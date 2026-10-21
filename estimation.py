import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import json
import threading
from typing import Optional

from csrnet_model import CSRNet


# ============================================================
# CAMERA AREA CONFIGURATION
# ============================================================

def get_supabase_client():
    import streamlit as st
    from supabase import create_client

    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )


def load_camera_config(camera_name):
    try:
        supabase = get_supabase_client()

        response = (
            supabase.table("camera_configs")
            .select("*")
            .eq("camera_name", camera_name)
            .execute()
        )

        if not response.data:
            return None

        config = response.data[0]

        return {
            "row1_area": config["row1_area"],
            "row2_area": config["row2_area"],
            "row3_area": config["row3_area"]
        }

    except Exception as e:
        print(f"Error loading camera configuration: {e}")
        raise


def save_camera_config(
    camera_name,
    row1_area,
    row2_area,
    row3_area
):
    supabase = get_supabase_client()

    supabase.table("camera_configs").upsert({
        "camera_name": camera_name,
        "row1_area": row1_area,
        "row2_area": row2_area,
        "row3_area": row3_area
    }).execute()
# ============================================================
# CROWD DENSITY ESTIMATOR
# ============================================================

class CrowdDensityEstimator:

    def __init__(
        self,
        device: str = "cpu",
        model_path: Optional[str] = None
    ):

        # ====================================================
        # DEVICE
        # ====================================================

        if (device == "cuda" and torch.cuda.is_available()):

            self.device = torch.device("cuda")

        else:

            self.device = torch.device( "cpu")


        # ====================================================
        # CSRNET
        # ====================================================

        self.model = CSRNet()


        if (
            model_path
            and os.path.exists(model_path)
        ):

            state_dict = torch.load(
                model_path,
                map_location=self.device
            )

            self.model.load_state_dict(
                state_dict,
                strict=True
            )

            print(
                f"Loaded CSRNet weights from {model_path}"
            )

        else:

            print(
                "Warning: No model weights found, "
                "using untrained CSRNet!"
            )


        self.model.to(
            self.device
        )

        self.model.eval()


        print(
            f"CSRNet model ready on device: "
            f"{self.device}"
        )


        # ====================================================
        # IMAGE PREPROCESSING
        # ====================================================

        self.transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])


    # ========================================================
    # PREPROCESS FRAME
    # ========================================================

    def preprocess_frame(
        self,
        frame: np.ndarray
    ) -> torch.Tensor:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frame_resized = cv2.resize(
            frame_rgb,
            (512, 512)
        )


        pil_image = Image.fromarray(
            frame_resized
        )


        tensor = self.transform(
            pil_image
        ).unsqueeze(0)


        return tensor.to(
            self.device
        )


    # ========================================================
    # ESTIMATE DENSITY
    # ========================================================

    def estimate_density(self,frame: np.ndarray):

        with torch.no_grad():

            input_tensor = (
                self.preprocess_frame(frame)
            )

            density_map = self.model(input_tensor)


            count = int(round(density_map.sum().item()))


            density_map_original = (
                density_map
                .squeeze()
                .cpu()
                .numpy()
            )


            h, w = frame.shape[:2]


            density_map_resized = cv2.resize(
                density_map_original,
                (w, h)
            )


            return (
                count,
                density_map_original,
                density_map_resized
            )


    # ========================================================
    # HEATMAP
    # ========================================================

    def visualize_density(
        self,
        frame,
        density_map,
        count
    ):

        h, w = frame.shape[:2]


        if density_map.shape != (
            h,
            w
        ):

            density_map_resized = cv2.resize(
                density_map,
                (w, h)
            )

        else:

            density_map_resized = density_map


        density_map_resized = cv2.GaussianBlur(density_map_resized,(5, 5),0)


        density_min = (
            density_map_resized.min())


        density_max = (
            density_map_resized.max()
        )


        density_normalized = (density_map_resized- density_min) / (density_max- density_min+ 1e-5)


        density_normalized = np.power(
            density_normalized,
            0.5
        )


        density_uint8 = (
            density_normalized
            * 255
        ).astype(
            np.uint8
        )


        heatmap = cv2.applyColorMap(
            density_uint8,
            cv2.COLORMAP_JET
        )


        overlay = cv2.addWeighted(
            frame,
            0.6,
            heatmap,
            0.4,
            0
        )


        return overlay


    # ========================================================
    # PHYSICAL REGIONS
    # ========================================================

    def create_physical_regions(
        self,
        frame,
        row1_area,
        row2_area,
        row3_area
    ):

        h, w = frame.shape[:2]


        # ====================================================
        # 3 ROWS x 3 COLUMNS = 9 ZONES
        # ====================================================

        area_matrix = [

            [
                row1_area,
                row1_area,
                row1_area
            ],

            [
                row2_area,
                row2_area,
                row2_area
            ],

            [
                row3_area,
                row3_area,
                row3_area
            ]
        ]


        x_points = [0,w // 3,(2 * w) // 3,w]
        y_points = [0,h // 3,(2 * h) // 3,h]

        regions = []

        region_number = 1


        for row in range(3):

            for col in range(3):

                x1 = x_points[col]
                x2 = x_points[col + 1]

                y1 = y_points[row]
                y2 = y_points[row + 1]


                polygon = np.array(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ],
                    dtype=np.int32
                )


                regions.append(
                    {
                        "number": region_number,
                        "polygon": polygon,
                        "area": area_matrix[row][col]
                    }
                )


                region_number += 1


        return regions


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify_regions(
        self,
        density_map,
        regions
    ):

        h, w = density_map.shape


        region_counts = []

        region_densities = []

        region_labels = []


        for region in regions:

            polygon = region["polygon"]

            area = region["area"]


            mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )


            cv2.fillPoly(
                mask,
                [polygon],
                1
            )


            regional_count = np.sum(
                density_map * mask
            )


            regional_count = max(
                0.0,
                float(regional_count)
            )


            # =================================================
            # N/A
            # =================================================

            if area is None:

                regional_density = None

                label = "N/A"


            # =================================================
            # CLASSIFY
            # =================================================

            else:

                regional_density = (regional_count / area)


                if regional_density < 0.40:

                    label = "FREE"

                elif regional_density <= 0.80:

                    label = "MODERATE"

                else:

                    label = "CONGESTED"


            region_counts.append(regional_count)
            region_densities.append(regional_density)
            region_labels.append(label)


        return (
            region_counts,
            region_densities,
            region_labels
        )


    # ========================================================
    # DRAW GRID
    # ========================================================

    def draw_region_grid(
        self,
        frame,
        regions,
        region_counts,
        region_densities,
        region_labels
    ):

        frame_copy = frame.copy()


        color_map = {

            "FREE": (
                0,
                255,
                0
            ),

            "MODERATE": (
                0,
                255,
                255
            ),

            "CONGESTED": (
                0,
                0,
                255
            ),

            "N/A": (
                128,
                128,
                128
            )
        }


        for i, region in enumerate(
            regions
        ):

            polygon = region["polygon"]

            label = region_labels[i]

            color = color_map[label]


            # =================================================
            # GRID BORDER
            # =================================================

            cv2.polylines(
                frame_copy,
                [polygon],
                True,
                color,
                3
            )


            # =================================================
            # COORDINATES
            # =================================================

            x1 = int(
                polygon[0][0]
            )

            x2 = int(
                polygon[1][0]
            )

            y1 = int(polygon[0][1])

            y2 = int(polygon[2][1])


            cx = int((x1 + x2) / 2)

            cy = int((y1 + y2) / 2)


            # =================================================
            # REGION NUMBER
            # =================================================

            number_text = str(
                region["number"]
            )


            number_size = cv2.getTextSize(
                number_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )[0]


            number_x = (
                cx
                - number_size[0] // 2)


            # =================================================
            # LABEL
            # =================================================

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.65

            thickness = 2


            text_size = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )[0]


            text_x = (
                cx
                - text_size[0] // 2
            )


            text_y = y1 + 32


            text_x = max(
                x1 + 5,
                min(
                    text_x,
                    x2 - text_size[0] - 5
                )
            )


            # =================================================
            # LABEL BACKGROUND
            # =================================================

            cv2.rectangle(
                frame_copy,
                (
                    text_x - 5,
                    text_y
                    - text_size[1]
                    - 7
                ),
                (
                    text_x
                    + text_size[0]
                    + 5,
                    text_y + 5
                ),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame_copy,
                label,
                (
                    text_x,
                    text_y
                ),
                font,
                font_scale,
                color,
                thickness
            )


            # =================================================
            # ZONE NUMBER
            # =================================================

            cv2.putText(
                frame_copy,
                number_text,
                (
                    number_x,
                    cy
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        return frame_copy


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    def process_video(
        self,
        video_path,
        output_path=None,
        show_preview=True,
        frame_skip=1,
        dashboard=None,
        row1_area=None,
        row2_area=None,
        row3_area=None
    ):

        cap = cv2.VideoCapture(
            video_path
        )


        if not cap.isOpened():

            raise ValueError(
                f"Could not open video: "
                f"{video_path}"
            )


        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        fps = int(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )


        if fps <= 0:
            fps = 30


        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )


        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        print(
            f"Video: {width}x{height}, "
            f"{fps} FPS, "
            f"{total_frames} frames"
        )


        # ====================================================
        # FRAME SKIP
        # ====================================================

        frame_skip = max(
            1,
            int(frame_skip)
        )


        # ====================================================
        # VIDEO WRITER
        # ====================================================

        writer = None


        if output_path:

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )


            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )


        # ====================================================
        # STATISTICS
        # ====================================================

        counts = []

        frame_count = 0

        processed_count = 0


        # These are cumulative statistics.
        # They are NOT displayed as the current
        # FREE / MODERATE / CONGESTED / N/A values.

        total_free = 0

        total_moderate = 0

        total_congested = 0

        total_na = 0


        # ====================================================
        # PROCESS FRAMES
        # ====================================================

        while True:

            ret, frame = cap.read()


            if not ret:
                break


            # =================================================
            # FRAME SKIP
            # =================================================

            if (
                frame_count
                % frame_skip
                == 0
            ):

                # =============================================
                # CSRNET
                # =============================================

                (
                    count,
                    density_map_original,
                    density_map
                ) = self.estimate_density(
                    frame
                )


                counts.append(
                    count
                )


                # =============================================
                # HEATMAP
                # =============================================

                vis_frame = (
                    self.visualize_density(
                        frame,
                        density_map,
                        count
                    )
                )


                # =============================================
                # CLASSIFICATION REGIONS
                #
                # Density map is used here because
                # classification must happen at the
                # density-map resolution.
                # =============================================

                classification_frame = np.zeros(
                    (
                        density_map_original.shape[0],
                        density_map_original.shape[1],
                        3
                    ),
                    dtype=np.uint8
                )


                classification_regions = (
                    self.create_physical_regions(
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
                ) = self.classify_regions(
                    density_map_original,
                    classification_regions
                )


                # =============================================
                # CURRENT FRAME ZONE ANALYSIS
                #
                # THESE ARE THE VALUES SENT TO THE GUI.
                #
                # They are recalculated every frame.
                # =============================================

                free = region_labels.count(
                    "FREE"
                )


                moderate = region_labels.count(
                    "MODERATE"
                )


                congested = region_labels.count(
                    "CONGESTED"
                )


                na = region_labels.count(
                    "N/A"
                )


                # =============================================
                # CUMULATIVE TOTALS
                #
                # These are only for final statistics.
                # They do NOT get sent as the live values.
                # =============================================

                total_free += free

                total_moderate += moderate

                total_congested += congested

                total_na += na


                # =============================================
                # DISPLAY REGIONS
                # =============================================

                display_regions = (
                    self.create_physical_regions(
                        frame,
                        row1_area,
                        row2_area,
                        row3_area
                    )
                )


                # =============================================
                # DRAW GRID
                # =============================================

                vis_frame = (
                    self.draw_region_grid(
                        vis_frame,
                        display_regions,
                        region_counts,
                        region_densities,
                        region_labels
                    )
                )


                # =============================================
                # VIDEO STATISTICS
                # =============================================

                average = (
                    float(
                        np.mean(counts)
                    )
                    if counts
                    else 0
                )


                maximum = (
                    float(
                        np.max(counts)
                    )
                    if counts
                    else 0
                )


                minimum = (
                    float(
                        np.min(counts)
                    )
                    if counts
                    else 0
                )


                # =============================================
                # DASHBOARD UPDATE
                # =============================================

                if dashboard is not None:

                    # -----------------------------------------
                    # Convert BGR -> RGB
                    # -----------------------------------------

                    rgb_frame = cv2.cvtColor(
                        vis_frame,
                        cv2.COLOR_BGR2RGB
                    )


                    # -----------------------------------------
                    # UPDATE VIDEO
                    # -----------------------------------------

                    dashboard.root.after(
                        0,
                        dashboard.update_video,
                        rgb_frame
                    )


                    # -----------------------------------------
                    # UPDATE LIVE ANALYSIS
                    #
                    # free/moderate/congested/na are CURRENT
                    # FRAME values.
                    # -----------------------------------------

                    dashboard.root.after(
                        0,
                        dashboard.update_analysis,
                        frame_count,
                        count,
                        free,
                        moderate,
                        congested,
                        na,
                        average,
                        maximum,
                        minimum
                    )


                # =============================================
                # OLD OPENCV PREVIEW
                # =============================================

                if (
                    show_preview
                    and dashboard is None
                ):

                    display_frame = cv2.resize(
                        vis_frame,
                        (1000, 700)
                    )


                    cv2.imshow(
                        "Crowd Density Estimation",
                        display_frame
                    )


                    if (
                        cv2.waitKey(1)
                        & 0xFF
                    ) == ord("q"):

                        break


                # =============================================
                # OUTPUT VIDEO
                # =============================================

                if writer:

                    writer.write(
                        vis_frame
                    )


                processed_count += 1


                if (
                    processed_count
                    % 10
                    == 0
                ):

                    print(
                        f"Processed "
                        f"{processed_count} "
                        f"frames..."
                    )


            frame_count += 1


        # ====================================================
        # RELEASE
        # ====================================================

        cap.release()


        if writer:
            writer.release()


        if (
            show_preview
            and dashboard is None
        ):

            cv2.destroyAllWindows()


        # ====================================================
        # FINAL STATISTICS
        # ====================================================

        stats = {

            "total_frames":
                total_frames,

            "processed_frames":
                processed_count,

            "avg_count":
                float(
                    np.mean(counts)
                )
                if counts
                else 0,

            "max_count":
                int(
                    np.max(counts)
                )
                if counts
                else 0,

            "min_count":
                int(
                    np.min(counts)
                )
                if counts
                else 0,

            "std_count":
                float(
                    np.std(counts)
                )
                if counts
                else 0,

            # ----------------------------------------------
            # CUMULATIVE ZONE STATISTICS
            # ----------------------------------------------

            "free_zones":
                total_free,

            "moderate_zones":
                total_moderate,

            "congested_zones":
                total_congested,

            "na_zones":
                total_na
        }


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print()

        print(
            "=========================================="
        )

        print(
            f"Average count: "
            f"{stats['avg_count']:.2f}"
        )

        print(
            f"Maximum count: "
            f"{stats['max_count']:.2f}"
        )

        print(
            f"Minimum count: "
            f"{stats['min_count']:.2f}"
        )

        print(
            "=========================================="
        )


        print()

        print(
            "CROWD ZONE ANALYSIS"
        )

        print(
            "=========================================="
        )

        print(
            f"FREE zones:       "
            f"{total_free}"
        )

        print(
            f"MODERATE zones:   "
            f"{total_moderate}"
        )

        print(
            f"CONGESTED zones:  "
            f"{total_congested}"
        )

        print(
            f"N/A zones:        "
            f"{total_na}"
        )

        print(
            "=========================================="
        )


        # ====================================================
        # FINAL DASHBOARD UPDATE
        # ====================================================

        if dashboard is not None:

            dashboard.root.after(
                0,
                dashboard.processing_finished,
                stats
            )


        return stats


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    def process_image(
        self,
        image_path,
        output_path=None
    ):

        frame = cv2.imread(
            image_path
        )


        if frame is None:

            raise ValueError(
                f"Could not load image: "
                f"{image_path}"
            )


        (
            count,
            density_map_original,
            density_map
        ) = self.estimate_density(
            frame
        )


        if output_path:

            camera_name = os.path.splitext(
                os.path.basename(
                    image_path
                )
            )[0]


            config = load_camera_config(
                camera_name
            )


            if config is None:

                raise RuntimeError(
                    "No camera area configuration found."
                )


            row1_area = config.get(
                "row1_area"
            )

            row2_area = config.get(
                "row2_area"
            )

            row3_area = config.get(
                "row3_area"
            )


            # ================================================
            # HEATMAP
            # ================================================

            vis_frame = (
                self.visualize_density(
                    frame,
                    density_map,
                    count
                )
            )


            # ================================================
            # DISPLAY REGIONS
            # ================================================

            regions = (
                self.create_physical_regions(
                    frame,
                    row1_area,
                    row2_area,
                    row3_area
                )
            )


            # ================================================
            # CLASSIFICATION REGIONS
            # ================================================

            classification_regions = (
                self.create_physical_regions(
                    np.zeros(
                        density_map_original.shape
                        + (3,),
                        dtype=np.uint8
                    ),
                    row1_area,
                    row2_area,
                    row3_area
                )
            )


            # ================================================
            # CLASSIFICATION
            # ================================================

            (
                region_counts,
                region_densities,
                region_labels
            ) = self.classify_regions(
                density_map_original,
                classification_regions
            )


            # ================================================
            # DRAW GRID
            # ================================================

            vis_frame = (
                self.draw_region_grid(
                    vis_frame,
                    regions,
                    region_counts,
                    region_densities,
                    region_labels
                )
            )


            # ================================================
            # SAVE
            # ================================================

            cv2.imwrite(
                output_path,
                vis_frame
            )


            print(
                f"Output saved to: "
                f"{output_path}"
            )


        return (
            count,
            density_map
        )


# ============================================================
# DASHBOARD PROCESSING
# ============================================================

def start_processing(
    video_path,
    row1_area,
    row2_area,
    row3_area,
    dashboard,
    frame_skip=1,
    output_path=None,
    model_path="models/CSRNet_SE.pth",
    device="cpu"
):

    def processing_thread():

        try:

            # =================================================
            # CREATE ESTIMATOR
            # =================================================

            estimator = CrowdDensityEstimator(
                device=device,
                model_path=model_path
            )


            # =================================================
            # PROCESS VIDEO
            # =================================================

            stats = estimator.process_video(
                video_path,
                output_path=output_path,
                show_preview=False,
                frame_skip=frame_skip,
                dashboard=dashboard,
                row1_area=row1_area,
                row2_area=row2_area,
                row3_area=row3_area
            )


            print(
                f"Video statistics: {stats}"
            )


        except Exception as e:

            print(
                "Processing error:",
                e
            )


            dashboard.root.after(
                0,
                dashboard.processing_error,
                e
            )


    # ========================================================
    # BACKGROUND THREAD
    # ========================================================

    thread = threading.Thread(
        target=processing_thread,
        daemon=True
    )


    thread.start()


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse
    import sys


    # ========================================================
    # CUSTOM ARGUMENT PARSER
    # ========================================================

    class CustomArgumentParser(
        argparse.ArgumentParser
    ):

        def error(
            self,
            message
        ):

            self.print_help()

            print(
                f"\nError: {message}"
            )

            sys.exit(1)


    parser = CustomArgumentParser(
        description="Crowd Density Estimation"
    )


    parser.add_argument(
        "--input",
        type=str,
        required=True
    )


    parser.add_argument(
        "--output",
        type=str,
        default=None
    )


    parser.add_argument(
        "--model",
        type=str,
        default="models/CSRNet_SE.pth"
    )


    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=[
            "cpu",
            "cuda"
        ]
    )


    parser.add_argument(
        "--no-preview",
        action="store_true"
    )


    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1
    )


    args = parser.parse_args()


    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not os.path.exists(
        args.input
    ):

        print(
            f"Input file not found: "
            f"{args.input}"
        )

        sys.exit(1)


    # ========================================================
    # VALIDATE FRAME SKIP
    # ========================================================

    if args.frame_skip < 1:

        print(
            "Frame skip must be at least 1."
        )

        sys.exit(1)


    # ========================================================
    # DETERMINE INPUT TYPE
    # ========================================================

    input_ext = os.path.splitext(
        args.input
    )[1].lower()


    # ========================================================
    # IMAGE
    # ========================================================

    if input_ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]:

        estimator = CrowdDensityEstimator(
            device=args.device,
            model_path=args.model
        )


        count, _ = estimator.process_image(
            args.input,
            args.output
        )


        print(
            f"Estimated crowd count: "
            f"{count:.2f}"
        )

        return


    # ========================================================
    # VIDEO + DASHBOARD
    # ========================================================

    from area_dashboard import start_dashboard


    def callback(video_path, row1, row2, row3, dashboard):
        start_processing(
        video_path=video_path,
        row1_area=row1,
        row2_area=row2,
        row3_area=row3,
        dashboard=dashboard,
        frame_skip=args.frame_skip,
        output_path=None,
        model_path=args.model,
        device=args.device
    )


    # ========================================================
    # START DASHBOARD
    # ========================================================

    start_dashboard(args.input,callback)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()