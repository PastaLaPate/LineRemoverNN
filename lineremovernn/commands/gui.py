import logging
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, messagebox

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from PIL import Image, ImageTk
from torch.amp.autocast_mode import autocast

from lineremovernn.commands.command import Command
from lineremovernn.model.lineremover import LineRemovalUNet
from lineremovernn.utils.consts import DEFAULT_MODELS, DEVICE
from lineremovernn.utils.saver import get_latest_model, load_model

logger = logging.getLogger("GUIInfer")


class GUIInferCommand(Command):
    def __init__(self):
        super().__init__(
            name="gui-infer",
            description="Open a graphical interface to remove lines from custom images.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-m",
            "--models",
            type=Path,
            default=DEFAULT_MODELS,
            help="Models location.",
        )

    def execute(self, args: Namespace) -> None:
        # 1. Initialize and load model
        model = LineRemovalUNet().to(DEVICE)
        lm = get_latest_model()
        if lm is not None:
            latest_model = load_model(lm[1], training=False)
            model.load_state_dict(latest_model.model_state)
            logger.info(f"Loaded weights from epoch {latest_model.stats.epoch}")
        else:
            logger.warning("No saved model found! Using random weights.")

        # Set to evaluation mode
        model.eval()

        # 2. Launch Tkinter Application Flow
        root = tk.Tk()
        LineRemoverApp(root, model)
        root.mainloop()


class LineRemoverApp:
    def __init__(self, root: tk.Tk, model: torch.nn.Module):
        self.root = root
        self.model = model

        self.root.title("Document Line Removal Tool")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f5f5f5")

        self.source_image: Image.Image | None = None
        self.processed_image: Image.Image | None = None

        # Transforms convert grayscale PIL to a [1, H, W] FloatTensor scaled to 0-1
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.to_pil = v2.ToPILImage()

        # --- UI Layout ---
        toolbar = tk.Frame(self.root, bg="#e0e0e0", height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = tk.Button(
            toolbar, text="Open Image", command=self.load_image, font=("Arial", 11)
        )
        self.btn_load.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_process = tk.Button(
            toolbar,
            text="Remove Lines",
            command=self.process_image,
            state=tk.DISABLED,
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
        )
        self.btn_process.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_save = tk.Button(
            toolbar,
            text="Save Result",
            command=self.save_image,
            state=tk.DISABLED,
            font=("Arial", 11),
        )
        self.btn_save.pack(side=tk.LEFT, padx=10, pady=10)

        workspace = tk.Frame(self.root, bg="#f5f5f5")
        workspace.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_input = tk.LabelFrame(
            workspace, text="Original Image", font=("Arial", 10, "bold"), bg="#f5f5f5"
        )
        frame_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lbl_input = tk.Label(frame_input, text="No image loaded", bg="#e8e8e8")
        self.lbl_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_output = tk.LabelFrame(
            workspace, text="Processed Image", font=("Arial", 10, "bold"), bg="#f5f5f5"
        )
        frame_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lbl_output = tk.Label(
            frame_output, text="Awaiting processing...", bg="#e8e8e8"
        )
        self.lbl_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            parent=self.root,
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")],
        )
        if not file_path:
            return

        try:
            # FIX 1: Convert to "L" (Grayscale / 1 Channel) instead of "RGB"
            self.source_image = Image.open(file_path).convert("L")
            self.processed_image = None
            self.lbl_output.config(image="", text="Awaiting processing...")
            self.btn_save.config(state=tk.DISABLED)

            self.display_preview(self.source_image, self.lbl_input)
            self.btn_process.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def process_image(self):
        if self.source_image is None:
            return

        self.root.config(cursor="watch")  # Fixed for Linux compatibility
        self.root.update()

        try:
            orig_w, orig_h = self.source_image.size

            # Now results in a [1, H, W] tensor
            img_tensor = self.transform(self.source_image)

            pad_h = (256 - orig_h % 256) % 256
            pad_w = (256 - orig_w % 256) % 256

            padded_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            _, ph, pw = padded_tensor.shape

            output_tensor = torch.zeros_like(padded_tensor)

            with torch.no_grad():
                for y in range(0, ph, 256):
                    for x in range(0, pw, 256):
                        tile = padded_tensor[:, y : y + 256, x : x + 256]
                        # Reshapes to [1, 1, 256, 256] for your Grayscale model
                        tile_batch = tile.unsqueeze(0).to(DEVICE)

                        with autocast(DEVICE):
                            pred_tile = self.model(tile_batch)

                        output_tensor[:, y : y + 256, x : x + 256] = pred_tile.squeeze(
                            0
                        ).cpu()

            # Crop back to original sizes
            output_tensor = output_tensor[:, :orig_h, :orig_w]

            # FIX 2: Replace any NaN or Inf values with safe defaults to stop the casting warning
            output_tensor = torch.nan_to_num(
                output_tensor, nan=0.0, posinf=1.0, neginf=0.0
            )
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

            # Convert back to PIL
            self.processed_image = self.to_pil(output_tensor)
            if self.processed_image:
                self.display_preview(self.processed_image, self.lbl_output)
            self.btn_save.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror(
                "Inference Error",
                f"An error occurred while cleaning the image:\n{str(e)}",
            )
        finally:
            self.root.config(cursor="")

    def save_image(self):
        if self.processed_image is None:
            return

        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
        )
        if not file_path:
            return

        try:
            self.processed_image.save(file_path)
            messagebox.showinfo("Success", "Cleaned image saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file:\n{str(e)}")

    def display_preview(self, pil_img: Image.Image, label_widget: tk.Label):
        max_w, max_h = 500, 550
        preview_img = pil_img.copy()
        preview_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(preview_img)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk  # type: ignore
