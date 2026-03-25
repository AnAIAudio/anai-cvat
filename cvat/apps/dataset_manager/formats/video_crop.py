# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Video crop export: extract labeled regions from videos as cropped clips, organized by label."""

import logging
import os
import subprocess

from cvat.apps.dataset_manager.util import make_zip_archive

from .registry import exporter

logger = logging.getLogger(__name__)


def _get_video_info(video_path):
    """Get video width, height, and fps using ffprobe."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-of', 'csv=p=0',
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    parts = result.stdout.strip().split(',')
    if len(parts) < 3:
        raise ValueError(f'Failed to probe video: {video_path}')
    width, height = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split('/'))
    fps = fps_num / fps_den
    return width, height, fps


def _build_crop_expression(keyframes, vid_w, vid_h, coord):
    """Build ffmpeg expression for dynamic crop position using keyframes with linear interpolation."""
    scale = vid_w if coord == 'x' else vid_h
    parts = []

    for i in range(len(keyframes) - 1):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]
        fa = kf_a['frame']
        fb = kf_b['frame']
        va = kf_a[coord] / 100 * scale
        vb = kf_b[coord] / 100 * scale

        if fa == fb:
            continue

        slope = (vb - va) / (fb - fa)
        parts.append(f'between(n\\,{fa}\\,{fb})*({va:.1f}+{slope:.4f}*(n-{fa}))')

    last_val = keyframes[-1][coord] / 100 * scale
    first_frame = keyframes[0]['frame']
    last_frame = keyframes[-1]['frame']
    fallback = (
        f'lt(n\\,{first_frame})*{keyframes[0][coord] / 100 * scale:.1f}'
        f'+gt(n\\,{last_frame})*{last_val:.1f}'
    )

    expr = '+'.join(parts) + '+' + fallback
    return expr


def _crop_video_by_bbox(video_path, keyframes, output_path, vid_w, vid_h, fps):
    """Crop video spatially (bbox) and temporally (label time range) using ffmpeg."""
    keyframes = sorted(keyframes, key=lambda s: s['frame'])

    t_start = keyframes[0]['time']
    t_end = keyframes[-1]['time']

    out_w = int(keyframes[0]['width'] / 100 * vid_w)
    out_h = int(keyframes[0]['height'] / 100 * vid_h)
    out_w += out_w % 2
    out_h += out_h % 2

    if len(keyframes) == 1:
        kf = keyframes[0]
        px = max(0, min(int(kf['x'] / 100 * vid_w), vid_w - out_w))
        py = max(0, min(int(kf['y'] / 100 * vid_h), vid_h - out_h))
        cmd = [
            'ffmpeg', '-y',
            '-ss', f'{t_start:.4f}',
            '-i', str(video_path),
            '-frames:v', '1',
            '-vf', f'crop={out_w}:{out_h}:{px}:{py}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-movflags', '+faststart',
            '-an',
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return proc.returncode == 0

    x_expr = _build_crop_expression(keyframes, vid_w, vid_h, 'x')
    y_expr = _build_crop_expression(keyframes, vid_w, vid_h, 'y')

    crop_filter = f'crop={out_w}:{out_h}:{x_expr}:{y_expr}'

    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{t_start:.4f}',
        '-to', f'{t_end:.4f}',
        '-i', str(video_path),
        '-vf', crop_filter,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-movflags', '+faststart',
        '-an',
        str(output_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        logger.warning('ffmpeg crop with expressions failed, trying fallback: %s', proc.stderr[-200:])
        avg_x = int(sum(kf['x'] for kf in keyframes) / len(keyframes) / 100 * vid_w)
        avg_y = int(sum(kf['y'] for kf in keyframes) / len(keyframes) / 100 * vid_h)
        avg_x = max(0, min(avg_x, vid_w - out_w))
        avg_y = max(0, min(avg_y, vid_h - out_h))

        cmd_fallback = [
            'ffmpeg', '-y',
            '-ss', f'{t_start:.4f}',
            '-to', f'{t_end:.4f}',
            '-i', str(video_path),
            '-vf', f'crop={out_w}:{out_h}:{avg_x}:{avg_y}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-movflags', '+faststart',
            '-an',
            str(output_path),
        ]
        proc2 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=300)
        if proc2.returncode != 0:
            logger.error('Video crop fallback also failed: %s', proc2.stderr[-200:])
            return False
    return True


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
    import json

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
        vid_w, vid_h, fps = _get_video_info(video_path)
    except ValueError as e:
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
