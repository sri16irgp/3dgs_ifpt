/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include "FFmpegVideoEncoder.hpp"

#ifndef HEADLESS
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#endif

#define QQ(rat) (rat.num / (double)rat.den)

#pragma warning(disable : 4996)

namespace sibr {

	bool FFVideoEncoder::ffmpegInitDone = false;

	FFVideoEncoder::FFVideoEncoder(
		const std::string& _filepath,
		double _fps,
		const sibr::Vector2i& size,
		bool forceResize
	) :
		filepath(_filepath),
		fps(_fps),
		_forceResize(forceResize)
	{
#ifndef HEADLESS

		/*
		 * av_register_all() is no longer required in modern FFmpeg.
		 * Codecs and formats are registered automatically.
		 */
		if (!ffmpegInitDone) {
			SIBR_LOG << "[FFMPEG] Initializing." << std::endl;
			ffmpegInitDone = true;
		}

		sibr::Vector2i sizeFix = size;
		bool hadToFix = false;

		if (sizeFix[0] % 2 != 0) {
			sizeFix[0] -= 1;
			hadToFix = true;
		}

		if (sizeFix[1] % 2 != 0) {
			sizeFix[1] -= 1;
			hadToFix = true;
		}

		if (hadToFix) {
			SIBR_WRG
				<< "Non-even video dimensions, resized to "
				<< sizeFix[0] << "x" << sizeFix[1] << "."
				<< std::endl;

			_forceResize = true;
		}

		init(sizeFix);

#endif
	}

	bool FFVideoEncoder::isFine() const
	{
		return initWasFine;
	}

	void FFVideoEncoder::close()
	{
#ifndef HEADLESS

		/*
		 * Flush remaining frames from encoder.
		 */
		if (pCodecCtx) {

			int ret = avcodec_send_frame(pCodecCtx, nullptr);

			if (ret >= 0) {

				while (true) {

					ret = avcodec_receive_packet(pCodecCtx, pkt);

					if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
						break;
					}

					if (ret < 0) {
						break;
					}

					av_packet_rescale_ts(
						pkt,
						pCodecCtx->time_base,
						video_st->time_base
					);

					pkt->stream_index = video_st->index;

					av_interleaved_write_frame(pFormatCtx, pkt);

					av_packet_unref(pkt);
				}
			}
		}

		if (pFormatCtx) {

			if (av_write_trailer(pFormatCtx) < 0) {
				SIBR_WRG
					<< "[FFMPEG] Could not write trailer."
					<< std::endl;
			}
		}

		if (pkt) {
			av_packet_free(&pkt);
		}

		if (frameYUV) {
			av_frame_free(&frameYUV);
		}

		if (pCodecCtx) {
			avcodec_free_context(&pCodecCtx);
		}

		if (pFormatCtx) {

			if (pFormatCtx->pb) {
				avio_closep(&pFormatCtx->pb);
			}

			avformat_free_context(pFormatCtx);
			pFormatCtx = nullptr;
		}

		video_st = nullptr;

		needFree = false;

#endif
	}

	FFVideoEncoder::~FFVideoEncoder()
	{
		if (needFree) {
			close();
		}
	}

	void FFVideoEncoder::init(const sibr::Vector2i& size)
	{
#ifndef HEADLESS

		w = size[0];
		h = size[1];

		const char* out_file = filepath.c_str();

		/*
		 * Modern FFmpeg returns const AVOutputFormat* and const AVCodec*.
		 *
		 * The existing SIBR header uses non-const pointers, therefore
		 * const_cast is used here to remain compatible with the existing
		 * class declaration.
		 */

		const AVOutputFormat* outputFormat =
			av_guess_format(nullptr, out_file, nullptr);

		if (!outputFormat) {
			SIBR_WRG
				<< "[FFMPEG] Could not determine output format."
				<< std::endl;
			return;
		}

		fmt = const_cast<AVOutputFormat*>(outputFormat);

		pFormatCtx = avformat_alloc_context();

		if (!pFormatCtx) {
			SIBR_WRG
				<< "[FFMPEG] Could not allocate format context."
				<< std::endl;
			return;
		}

		pFormatCtx->oformat = outputFormat;

		const bool isH264 =
			pFormatCtx->oformat->video_codec == AV_CODEC_ID_H264;

		if (isH264) {
			SIBR_LOG
				<< "[FFMPEG] Found H264 codec."
				<< std::endl;
		}
		else {
			SIBR_LOG
				<< "[FFMPEG] Found codec with ID "
				<< pFormatCtx->oformat->video_codec
				<< " (not H264)."
				<< std::endl;
		}

		if (!(pFormatCtx->oformat->flags & AVFMT_NOFILE)) {

			if (
				avio_open(
					&pFormatCtx->pb,
					out_file,
					AVIO_FLAG_WRITE
				) < 0
			) {
				SIBR_WRG
					<< "[FFMPEG] Could not open file "
					<< filepath
					<< std::endl;

				return;
			}
		}

		const AVCodec* encoder =
			avcodec_find_encoder(
				pFormatCtx->oformat->video_codec
			);

		if (!encoder) {
			SIBR_WRG
				<< "[FFMPEG] Could not find codec."
				<< std::endl;
			return;
		}

		pCodec = const_cast<AVCodec*>(encoder);

		video_st =
			avformat_new_stream(
				pFormatCtx,
				nullptr
			);

		if (!video_st) {
			SIBR_WRG
				<< "[FFMPEG] Could not create stream."
				<< std::endl;
			return;
		}

		pCodecCtx =
			avcodec_alloc_context3(encoder);

		if (!pCodecCtx) {
			SIBR_WRG
				<< "[FFMPEG] Could not allocate codec context."
				<< std::endl;
			return;
		}

		pCodecCtx->codec_id =
			pFormatCtx->oformat->video_codec;

		pCodecCtx->codec_type =
			AVMEDIA_TYPE_VIDEO;

		pCodecCtx->pix_fmt =
			AV_PIX_FMT_YUV420P;

		pCodecCtx->width = w;
		pCodecCtx->height = h;

		pCodecCtx->gop_size = 10;

		const int roundedFPS =
			std::max(1, static_cast<int>(std::round(fps)));

		pCodecCtx->time_base =
			AVRational{1, roundedFPS};

		pCodecCtx->framerate =
			AVRational{roundedFPS, 1};

		video_st->time_base =
			pCodecCtx->time_base;

		/*
		 * Required for containers that store global codec headers.
		 */
		if (
			pFormatCtx->oformat->flags &
			AVFMT_GLOBALHEADER
		) {
			pCodecCtx->flags |=
				AV_CODEC_FLAG_GLOBAL_HEADER;
		}

		AVDictionary* param = nullptr;

		if (
			pCodecCtx->codec_id ==
			AV_CODEC_ID_H264
		) {

			av_dict_set(
				&param,
				"preset",
				"slow",
				0
			);

			av_dict_set(
				&param,
				"tune",
				"zerolatency",
				0
			);
		}

		int res =
			avcodec_open2(
				pCodecCtx,
				encoder,
				&param
			);

		av_dict_free(&param);

		if (res < 0) {

			SIBR_WRG
				<< "[FFMPEG] Failed to open encoder, error: "
				<< res
				<< std::endl;

			return;
		}

		/*
		 * Copy codec information into AVStream.
		 *
		 * video_st->codec no longer exists in modern FFmpeg.
		 */
		res =
			avcodec_parameters_from_context(
				video_st->codecpar,
				pCodecCtx
			);

		if (res < 0) {

			SIBR_WRG
				<< "[FFMPEG] Failed to copy codec parameters."
				<< std::endl;

			return;
		}

		video_st->time_base =
			pCodecCtx->time_base;

		av_dump_format(
			pFormatCtx,
			0,
			out_file,
			1
		);

		res =
			avformat_write_header(
				pFormatCtx,
				nullptr
			);

		if (res < 0) {

			SIBR_WRG
				<< "[FFMPEG] Failed to write video header."
				<< std::endl;

			return;
		}

		/*
		 * Prepare YUV frame.
		 */
		frameYUV = av_frame_alloc();

		if (!frameYUV) {

			SIBR_WRG
				<< "[FFMPEG] Could not allocate frame."
				<< std::endl;

			return;
		}

		frameYUV->format =
			static_cast<int>(
				pCodecCtx->pix_fmt
			);

		frameYUV->width = w;
		frameYUV->height = h;

		/*
		 * Data is backed by the OpenCV YUV buffer later, so
		 * we preserve the original manual plane layout.
		 */
		frameYUV->linesize[0] = w;
		frameYUV->linesize[1] = w / 2;
		frameYUV->linesize[2] = w / 2;

		yuSize[0] =
			frameYUV->linesize[0] * h;

		yuSize[1] =
			frameYUV->linesize[1] * h / 2;

		pkt = av_packet_alloc();

		if (!pkt) {

			SIBR_WRG
				<< "[FFMPEG] Could not allocate packet."
				<< std::endl;

			return;
		}

		frameCount = 0;

		initWasFine = true;
		needFree = true;

#endif
	}

	bool FFVideoEncoder::operator<<(cv::Mat frame)
	{
#ifndef HEADLESS

		if (!video_st || !pCodecCtx) {
			return false;
		}

		cv::Mat local;

		if (
			frame.cols != w ||
			frame.rows != h
		) {

			if (_forceResize) {

				cv::resize(
					frame,
					local,
					cv::Size(w, h)
				);

			}
			else {

				SIBR_WRG
					<< "[FFMPEG] Frame doesn't have the same "
					<< "dimensions as the video."
					<< std::endl;

				return false;
			}

		}
		else {
			local = frame;
		}

		cv::cvtColor(
			local,
			cvFrameYUV,
			cv::COLOR_BGR2YUV_I420
		);

		frameYUV->data[0] =
			cvFrameYUV.data;

		frameYUV->data[1] =
			frameYUV->data[0] +
			yuSize[0];

		frameYUV->data[2] =
			frameYUV->data[1] +
			yuSize[1];

		/*
		 * Since codec time_base is 1/fps, one frame corresponds
		 * directly to one PTS increment.
		 */
		frameYUV->pts =
			static_cast<int64_t>(
				frameCount
			);

		++frameCount;

		return encode(frameYUV);

#else

		SIBR_ERR
			<< "Not supported in headless"
			<< std::endl;

		return false;

#endif
	}

	bool FFVideoEncoder::operator<<(
		const sibr::ImageRGB& frame
	)
	{
		return (*this) << frame.toOpenCVBGR();
	}

#ifndef HEADLESS

	bool FFVideoEncoder::encode(AVFrame* frame)
	{
		/*
		 * Modern FFmpeg encoding API:
		 *
		 * avcodec_send_frame()
		 *         ↓
		 * avcodec_receive_packet()
		 */

		int ret =
			avcodec_send_frame(
				pCodecCtx,
				frame
			);

		if (ret < 0) {

			SIBR_WRG
				<< "[FFMPEG] Failed to send frame to encoder."
				<< std::endl;

			return false;
		}

		while (true) {

			ret =
				avcodec_receive_packet(
					pCodecCtx,
					pkt
				);

			if (
				ret == AVERROR(EAGAIN) ||
				ret == AVERROR_EOF
			) {
				break;
			}

			if (ret < 0) {

				SIBR_WRG
					<< "[FFMPEG] Failed to receive encoded packet."
					<< std::endl;

				return false;
			}

			/*
			 * Convert encoder timestamps to stream timestamps.
			 */
			av_packet_rescale_ts(
				pkt,
				pCodecCtx->time_base,
				video_st->time_base
			);

			pkt->stream_index =
				video_st->index;

			ret =
				av_interleaved_write_frame(
					pFormatCtx,
					pkt
				);

			av_packet_unref(pkt);

			if (ret < 0) {

				SIBR_WRG
					<< "[FFMPEG] Failed to write encoded packet."
					<< std::endl;

				return false;
			}
		}

		return true;
	}

#endif

}