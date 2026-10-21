
import os
import tempfile

import cv2
import streamlit as st
import numpy as np

from estimation import (
    CrowdDensityEstimator,
    load_camera_config,
    save_camera_config
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join("models", "CSRNet_SE.pth")

st.set_page_config(
    page_title="CCTV Crowd Analytics",
    page_icon="📹",
    layout="wide"
)


# ============================================================
# AREA CONFIGURATION
# ============================================================

def get_saved_area(saved, key):
    if key not in saved:
        return 100.0

    value = saved[key]

    if value is None:
        return 0.0

    return float(value)


# ============================================================
# WEBSITE
# ============================================================

st.title("CCTV-Based Crowd Analytics")

st.write(
    "Upload surveillance footage to estimate crowd density "
    "and identify free, moderate, and congested zones."
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_file is not None:

    camera_name = os.path.splitext(
        os.path.basename(uploaded_file.name)
    )[0]

    try:
        saved = load_camera_config(camera_name) or {}
    except Exception as error:
        st.error(
            f"Could not load saved area configuration: {error}"
        )
        st.stop()

    st.subheader("Configure physical areas")

    col1, col2, col3 = st.columns(3)

    with col1:
        row1_area = st.number_input(
            "Top row area",
            min_value=0.0,
            value=get_saved_area(saved, "row1_area"),
            step=10.0
        )

    with col2:
        row2_area = st.number_input(
            "Middle row area",
            min_value=0.0,
            value=get_saved_area(saved, "row2_area"),
            step=10.0
        )

    with col3:
        row3_area = st.number_input(
            "Bottom row area",
            min_value=0.0,
            value=get_saved_area(saved, "row3_area"),
            step=10.0
        )

    if st.button("Save Area Configuration"):
        try:
            save_camera_config(
                camera_name,
                None if row1_area == 0 else row1_area,
                None if row2_area == 0 else row2_area,
                None if row3_area == 0 else row3_area
            )

            st.success("Area configuration saved.")

        except Exception as error:
            st.error(
                f"Could not save area configuration: {error}"
            )

    # ========================================================
    # FRAME SKIP OPTION
    # ========================================================

    frame_skip = st.number_input(
        "Frame Skip",
        min_value=1,
        value=10,
        step=1,
        help=(
            "1 processes every frame. "
            "10 processes every 10th frame. "
            "Higher values generally make analysis faster."
        )
    )

    if st.button("Start Crowd Analysis", type="primary"):

        row1_area = None if row1_area == 0 else row1_area
        row2_area = None if row2_area == 0 else row2_area
        row3_area = None if row3_area == 0 else row3_area

        try:
            save_camera_config(
                camera_name,
                row1_area,
                row2_area,
                row3_area
            )
        except Exception as error:
            st.error(
                f"Could not save area configuration: {error}"
            )
            st.stop()

        progress = st.progress(0)
        status = st.empty()
        video_display = st.empty()

        temp_input = None
        temp_output = None

        try:
            # ------------------------------------------------
            # SAVE UPLOADED VIDEO TEMPORARILY
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_input = temp_file.name

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_output = output_file.name
            output_file.close()

            # ------------------------------------------------
            # INITIALIZE MODEL AND VIDEO
            # ------------------------------------------------

            estimator = CrowdDensityEstimator(
                device="cpu",
                model_path=MODEL_PATH
            )

            cap = cv2.VideoCapture(temp_input)

            if not cap.isOpened():
                raise ValueError(
                    "Could not open the uploaded video."
                )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if width <= 0 or height <= 0:
                cap.release()

                raise ValueError(
                    "Could not read the video's dimensions."
                )

            writer = cv2.VideoWriter(
                temp_output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not writer.isOpened():
                cap.release()

                raise RuntimeError(
                    "Could not create the output video."
                )

            # ------------------------------------------------
            # INITIALIZE STATISTICS
            # ------------------------------------------------

            processed_frames = 0
            counts = []

            total_free = 0
            total_moderate = 0
            total_congested = 0
            total_na = 0

            frame_number = 0
            last_output_frame = None

            # ------------------------------------------------
            # PROCESS VIDEO WITH FRAME SKIP
            # ------------------------------------------------

            try:
                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Process only selected frames.
                    if frame_number % int(frame_skip) == 0:

                        count, density_original, density_map = (
                            estimator.estimate_density(frame)
                        )

                        counts.append(count)

                        heatmap_frame = (
                            estimator.visualize_density(
                                frame,
                                density_map,
                                count
                            )
                        )

                        classification_frame = np.zeros(
                            (
                                density_original.shape[0],
                                density_original.shape[1],
                                3
                            ),
                            dtype=np.uint8
                        )

                        classification_regions = (
                            estimator.create_physical_regions(
                                classification_frame,
                                row1_area,
                                row2_area,
                                row3_area
                            )
                        )

                        (
                            region_counts,
                            region_densities,
                            region_labels
                        ) = estimator.classify_regions(
                            density_original,
                            classification_regions
                        )

                        total_free += region_labels.count("FREE")
                        total_moderate += region_labels.count("MODERATE")
                        total_congested += region_labels.count("CONGESTED")
                        total_na += region_labels.count("N/A")

                        display_regions = (
                            estimator.create_physical_regions(
                                frame,
                                row1_area,
                                row2_area,
                                row3_area
                            )
                        )

                        output_frame = estimator.draw_region_grid(
                            heatmap_frame,
                            display_regions,
                            region_counts,
                            region_densities,
                            region_labels
                        )

                        last_output_frame = output_frame.copy()
                        processed_frames += 1

                        # ------------------------------------
                        # LIVE ANALYSED FRAME PREVIEW
                        # ------------------------------------

                        preview_width = min(720, width)

                        preview_height = max(
                            1,
                            int(height * preview_width / width)
                        )

                        preview_frame = cv2.resize(
                            output_frame,
                            (preview_width, preview_height)
                        )

                        preview = cv2.cvtColor(
                            preview_frame,
                            cv2.COLOR_BGR2RGB
                        )

                        video_display.image(
                            preview,
                            caption="Live Analysed Video"
                        )

                        status.write(
                            f"Analysed frame {frame_number + 1} "
                            f"of {total_frames if total_frames > 0 else '?'}"
                            f" | Crowd count: {count}"
                        )

                    # ----------------------------------------
                    # WRITE OUTPUT AT ORIGINAL FRAME RATE
                    # ----------------------------------------

                    # Reuse the latest analysed frame for skipped
                    # frames so the output keeps its original
                    # duration and frame rate.
                    if last_output_frame is not None:
                        writer.write(last_output_frame)

                    frame_number += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_number / total_frames, 1.0)
                        )

            finally:
                cap.release()
                writer.release()

            # ------------------------------------------------
            # CHECK PROCESSING RESULTS
            # ------------------------------------------------

            if processed_frames == 0:
                raise ValueError(
                    "The uploaded video contains no readable frames."
                )

            progress.progress(1.0)
            status.success("Crowd analysis completed.")

            # ------------------------------------------------
            # LOAD COMPLETED OUTPUT VIDEO
            # ------------------------------------------------

            with open(temp_output, "rb") as video_file:
                output_video_bytes = video_file.read()

            # ------------------------------------------------
            # ANALYSIS RESULTS
            # ------------------------------------------------

            st.subheader("Analysis Results")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Average crowd count",
                f"{np.mean(counts):.2f}"
            )

            c2.metric(
                "Maximum crowd count",
                int(np.max(counts))
            )

            c3.metric(
                "Minimum crowd count",
                int(np.min(counts))
            )

            c4.metric(
                "Frames analysed",
                processed_frames
            )

            st.subheader("Cumulative Zone Statistics")

            a, b, c, d = st.columns(4)

            a.metric("FREE", total_free)
            b.metric("MODERATE", total_moderate)
            c.metric("CONGESTED", total_congested)
            d.metric("N/A", total_na)

            # ------------------------------------------------
            # DOWNLOAD OUTPUT VIDEO
            # ------------------------------------------------

            st.download_button(
                "Download Analysed Video",
                data=output_video_bytes,
                file_name=f"{camera_name}_analysed.mp4",
                mime="video/mp4"
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

        finally:
            for path in (temp_input, temp_output):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

else:
    st.info("Upload a video to begin.")
