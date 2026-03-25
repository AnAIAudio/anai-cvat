# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Video crop export: extract labeled regions from videos as cropped clips, organized by label.

Uses PyAV (libav) instead of ffmpeg CLI since CVAT containers don't include
ffmpeg/ffprobe binaries (only the shared libraries for PyAV).
"""

import json
import logging
import os

import av
import numpy as np

from cvat.apps.dataset_manager.util import make_zip_archive

from .registry import exporter

logger = logging.getLogger(__name__)


def _get_video_info(video_path):
    """Get video width, height, fps, and total frames using PyAV."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        total_frames = stream.frames or 0
        return width, height, fps, total_frames


def _interpolate_bbox(keyframes, frame_number):
    """Linearly interpolate bbox at a given frame number between keyframes."""
    if not keyframes:
        return None

    # Before first keyframe
    if frame_number <= keyframes[0]['frame']:
        return keyframes[0]
    # After last keyframe
    if frame_number >= keyframes[-1]['frame']:
        return keyframes[-1]

    # Find surrounding keyframes
    for i in range(len(keyframes) - 1):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]
        if kf_a['frame'] <= frame_number <= kf_b['frame']:
            if kf_a['frame'] == kf_b['frame']:
                return kf_a
            t = (frame_number - kf_a['frame']) / (kf_b['frame'] - kf_a['frame'])
            return {
                'frame': frame_number,
                'x': kf_a['x'] + t * (kf_b['x'] - kf_a['x']),
                'y': kf_a['y'] + t * (kf_b['y'] - kf_a['y']),
                'width': kf_a['width'] + t * (kf_b['width'] - kf_a['width']),
                'height': kf_a['height'] + t * (kf_b['height'] - kf_a['height']),
            }

    return keyframes[-1]


def _crop_video_by_bbox(video_path, keyframes, output_path, vid_w, vid_h, fps):
    """Crop video spatially (bbox) and temporally using PyAV."""
    keyframes = sorted(keyframes, key=lambda s: s['frame'])

    start_frame = keyframes[0]['frame']
    end_frame = keyframes[-1]['frame']

    # Output size from first keyframe bbox
    out_w = int(keyframes[0]['width'] / 100 * vid_w)
    out_h = int(keyframes[0]['height'] / 100 * vid_h)
    out_w += out_w % 2  # ensure even for h264
    out_h += out_h % 2

    if out_w <= 0 or out_h <= 0:
        logger.warning('Invalid crop dimensions: %dx%d', out_w, out_h)
        return False

    try:
        input_container = av.open(str(video_path))
        output_container = av.open(str(output_path), mode='w')

        in_stream = input_container.streams.video[0]
        out_stream = output_container.add_stream('h264', rate=fps)
        out_stream.width = out_w
        out_stream.height = out_h
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'preset': 'fast', 'crf': '23', 'movflags': '+faststart'}

        frame_count = 0
        for frame in input_container.decode(video=0):
            frame_num = frame_count
            frame_count += 1

            if frame_num < start_frame:
                continue
            if frame_num > end_frame:
                break

            # Get interpolated bbox for this frame
            bbox = _interpolate_bbox(keyframes, frame_num)
            if bbox is None:
                continue

            # Convert percentage to pixel coordinates
            px = max(0, int(bbox['x'] / 100 * vid_w))
            py = max(0, int(bbox['y'] / 100 * vid_h))

            # Clamp to video bounds
            px = min(px, vid_w - out_w)
            py = min(py, vid_h - out_h)
            px = max(0, px)
            py = max(0, py)

            # Convert frame to numpy, crop, and encode
            img = frame.to_ndarray(format='rgb24')
            cropped = img[py:py + out_h, px:px + out_w]

            # Handle edge cases where crop region exceeds image
            if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
                padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                h = min(cropped.shape[0], out_h)
                w = min(cropped.shape[1], out_w)
                padded[:h, :w] = cropped[:h, :w]
                cropped = padded

            out_frame = av.VideoFrame.from_ndarray(cropped, format='rgb24')
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)

        # Flush encoder
        for packet in out_stream.encode():
            output_container.mux(packet)

        output_container.close()
        input_container.close()
        return True

    except Exception:
        logger.exception('Failed to crop video')
        return False


def _convert_track_to_keyframes(track, vid_w, vid_h, fps):
    """Convert CVAT Track (with TrackedShapes) to keyframe sequence for video crop.

    CVAT TrackedShape.points = [x1, y1, x2, y2] (absolute pixels, for rectangle type)
    Video crop keyframes use percentage coordinates (0-100).
    """
    keyframes = []
    for shape in track.shapes:
        if shape.outside:
            continue
        if shape.type != 'rectangle':
            continue

        points = shape.points
        if len(points) < 4:
            continue

        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]

        keyframes.append({
            'frame': shape.frame,
            'x': x1 / vid_w * 100,
            'y': y1 / vid_h * 100,
            'width': (x2 - x1) / vid_w * 100,
            'height': (y2 - y1) / vid_h * 100,
            'time': shape.frame / fps,
        })

    return sorted(keyframes, key=lambda k: k['frame'])


def _export_video_crop(dst_file, temp_dir, instance_data, **options):
    db_data = instance_data._db_data

    if not hasattr(db_data, 'video'):
        logger.warning('Video crop export: task does not contain video data')
        make_zip_archive(temp_dir, dst_file)
        return

    video_path = db_data.get_raw_data_dirname() / db_data.video.path
    if not video_path.is_file():
        logger.warning('Video crop export: video file not found: %s', video_path)
        make_zip_archive(temp_dir, dst_file)
        return

    try:
        vid_w, vid_h, fps, _ = _get_video_info(video_path)
    except Exception as e:
        logger.warning('Video crop export: failed to probe video: %s', e)
        make_zip_archive(temp_dir, dst_file)
        return

    exported_count = 0
    labels_data = []

    for track in instance_data.tracks:
        label_str = track.label or 'unlabeled'

        keyframes = _convert_track_to_keyframes(track, vid_w, vid_h, fps)
        if not keyframes:
            continue

        label_dir = os.path.join(temp_dir, label_str)
        os.makedirs(label_dir, exist_ok=True)

        track_id = track.id if track.id is not None else exported_count
        output_filename = f'track_{track_id}.mp4'
        output_path = os.path.join(label_dir, output_filename)

        logger.info('Cropping video for track %s, label=%s', track_id, label_str)
        success = _crop_video_by_bbox(video_path, keyframes, output_path, vid_w, vid_h, fps)

        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            exported_count += 1
            labels_data.append({
                'track_id': track_id,
                'label': label_str,
                'video': f'{label_str}/{output_filename}',
                'keyframes': len(keyframes),
                'time_start': min(kf['time'] for kf in keyframes),
                'time_end': max(kf['time'] for kf in keyframes),
            })
        else:
            logger.warning('Failed to crop video for track %s', track_id)

    with open(os.path.join(temp_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    logger.info('Video crop export complete: %s clips exported', exported_count)
    make_zip_archive(temp_dir, dst_file)


@exporter(name='Video Crop', version='1.0', ext='ZIP', display_name='Video Crop {VERSION}')
def _export(dst_file, temp_dir, instance_data, **options):
    _export_video_crop(dst_file, temp_dir, instance_data, **options)
