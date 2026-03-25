// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import './styles.scss';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';

import { CombinedState } from 'reducers';
import { changeFrameAsync } from 'actions/annotation-actions';

function VideoPlayer(): JSX.Element {
    const videoRef = useRef<HTMLVideoElement>(null);
    const isSeeking = useRef(false);
    const [fps, setFps] = useState<number>(0);

    const dispatch = useDispatch();

    const {
        jobId,
        frame,
        startFrame,
        stopFrame,
        totalFrames,
    } = useSelector((state: CombinedState) => {
        const job = state.annotation.job.instance!;
        return {
            jobId: job.id,
            frame: state.annotation.player.frame.number,
            startFrame: job.startFrame,
            stopFrame: job.stopFrame,
            totalFrames: job.stopFrame - job.startFrame + 1,
        };
    }, shallowEqual);

    const videoSrc = `/api/jobs/${jobId}/data?type=video`;

    // Calculate fps from video duration once loaded
    const handleLoadedMetadata = useCallback(() => {
        const video = videoRef.current;
        if (video && video.duration > 0 && totalFrames > 0) {
            setFps(totalFrames / video.duration);
        }
    }, [totalFrames]);

    // Sync video time when frame changes externally (e.g., from player controls)
    useEffect(() => {
        const video = videoRef.current;
        if (video && !isSeeking.current && fps > 0) {
            const targetTime = (frame - startFrame) / fps;
            if (Math.abs(video.currentTime - targetTime) > 0.05) {
                video.currentTime = targetTime;
            }
        }
    }, [frame, startFrame, fps]);

    // When user seeks in the video, update the frame number
    const handleSeeked = useCallback(() => {
        const video = videoRef.current;
        if (video && fps > 0) {
            const targetFrame = Math.round(video.currentTime * fps) + startFrame;
            const clampedFrame = Math.max(startFrame, Math.min(stopFrame, targetFrame));
            if (clampedFrame !== frame) {
                isSeeking.current = true;
                dispatch(changeFrameAsync(clampedFrame)).then(() => {
                    isSeeking.current = false;
                });
            }
        }
    }, [dispatch, fps, startFrame, stopFrame, frame]);

    // On timeupdate during playback, sync frame
    const handleTimeUpdate = useCallback(() => {
        const video = videoRef.current;
        if (video && !video.paused && fps > 0) {
            const targetFrame = Math.round(video.currentTime * fps) + startFrame;
            const clampedFrame = Math.max(startFrame, Math.min(stopFrame, targetFrame));
            if (clampedFrame !== frame) {
                isSeeking.current = true;
                dispatch(changeFrameAsync(clampedFrame)).then(() => {
                    isSeeking.current = false;
                });
            }
        }
    }, [dispatch, fps, startFrame, stopFrame, frame]);

    return (
        <div className='cvat-video-player-wrapper'>
            <video
                ref={videoRef}
                className='cvat-video-player'
                src={videoSrc}
                controls
                onLoadedMetadata={handleLoadedMetadata}
                onSeeked={handleSeeked}
                onTimeUpdate={handleTimeUpdate}
            />
        </div>
    );
}

export default React.memo(VideoPlayer);
