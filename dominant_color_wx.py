import wx
from PIL import Image, ImageFilter
import numpy as np


def dominant_color_histogram(path, bins_per_channel=32, blur=True):
    img = Image.open(path).convert("RGB")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

    arr = np.array(img, dtype=np.uint8)
    pixels = arr.reshape(-1, 3)

    factor = 256 / bins_per_channel
    r_bins = (pixels[:, 0] / factor).astype(int)
    g_bins = (pixels[:, 1] / factor).astype(int)
    b_bins = (pixels[:, 2] / factor).astype(int)

    r_bins = np.clip(r_bins, 0, bins_per_channel - 1)
    g_bins = np.clip(g_bins, 0, bins_per_channel - 1)
    b_bins = np.clip(b_bins, 0, bins_per_channel - 1)

    bin_index = (
        r_bins * (bins_per_channel ** 2)
        + g_bins * bins_per_channel
        + b_bins
    )

    unique_bins, counts = np.unique(bin_index, return_counts=True)
    dom_bin = unique_bins[np.argmax(counts)]

    rb = dom_bin // (bins_per_channel ** 2)
    gb = (dom_bin // bins_per_channel) % bins_per_channel
    bb = dom_bin % bins_per_channel

    mask = (r_bins == rb) & (g_bins == gb) & (b_bins == bb)
    cluster_pixels = pixels[mask]

    dominant_rgb = cluster_pixels.mean(axis=0)
    return tuple(int(x) for x in dominant_rgb)


class DominantColorFrame(wx.Frame):
    def __init__(self):
        super().__init__(
            None,
            title="Dominant Color Finder",
            size=(720, 600),
            style=wx.DEFAULT_FRAME_STYLE
        )

        # --- ensure window comes to front ---
        wx.CallAfter(self.Raise)
        wx.CallAfter(self.SetFocus)

        self.image_path = None

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="Dominant Color Finder")
        title.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vbox.Add(title, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        # --- image preview area ---
        self.preview_panel = wx.Panel(panel, size=(320, 320))
        self.preview_panel.SetBackgroundColour(wx.Colour(230, 230, 230))

        self.bitmap_ctrl = wx.StaticBitmap(self.preview_panel)
        preview_sizer = wx.BoxSizer(wx.VERTICAL)
        preview_sizer.Add(self.bitmap_ctrl, flag=wx.ALIGN_CENTER | wx.ALL, border=5)
        self.preview_panel.SetSizer(preview_sizer)

        vbox.Add(self.preview_panel, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        # --- controls ---
        controls = wx.BoxSizer(wx.HORIZONTAL)

        open_btn = wx.Button(panel, label="Open Image")
        open_btn.Bind(wx.EVT_BUTTON, self.on_open)
        controls.Add(open_btn, flag=wx.RIGHT, border=8)

        controls.Add(wx.StaticText(panel, label="Bins:"), flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        self.bin_ctrl = wx.SpinCtrl(panel, min=4, max=128, initial=32)
        controls.Add(self.bin_ctrl, flag=wx.RIGHT, border=10)

        self.blur_ctrl = wx.CheckBox(panel, label="Gaussian Blur")
        self.blur_ctrl.SetValue(True)
        controls.Add(self.blur_ctrl, flag=wx.RIGHT, border=10)

        analyze_btn = wx.Button(panel, label="Analyze")
        analyze_btn.Bind(wx.EVT_BUTTON, self.on_analyze)
        controls.Add(analyze_btn, flag=wx.RIGHT, border=8)

        help_btn = wx.Button(panel, label="Help")
        help_btn.Bind(wx.EVT_BUTTON, self.on_help)
        controls.Add(help_btn)

        vbox.Add(controls, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        self.result_text = wx.StaticText(panel, label="")
        self.result_text.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vbox.Add(self.result_text, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        self.color_panel = wx.Panel(panel, size=(120, 120))
        self.color_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        vbox.Add(self.color_panel, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        panel.SetSizer(vbox)
        self.Centre()

    def on_open(self, event):
        with wx.FileDialog(
            self,
            "Open Image",
            wildcard="Image files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return

            self.image_path = dialog.GetPath()

        # --- load & scale image while preserving aspect ratio ---
        img = wx.Image(self.image_path)
        iw, ih = img.GetSize()

        max_w, max_h = 300, 300
        scale = min(max_w / iw, max_h / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)

        img = img.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
        self.bitmap_ctrl.SetBitmap(wx.Bitmap(img))
        self.preview_panel.Layout()

        self.result_text.SetLabel("")
        self.color_panel.SetBackgroundColour(wx.Colour(255, 255, 255))

    def on_help(self, event):
        help_text = """Dominant Color Finder Help

What this program does:
Finds the most dominant RGB color in an image using histogram binning.
No machine learning or resizing is used for analysis.

Controls:

Open Image:
  Select an image file. The preview is scaled for display only.

Bins:
  Controls color resolution.
  Typical values:
    24–32 for photos
    32–64 for logos

Gaussian Blur:
  Reduces noise and JPEG artifacts before analysis.

Analyze:
  Computes the dominant color.

Results:
  Shows the RGB tuple and a color preview square.

Notes:
• Full-resolution image is analyzed
• No cropping or resizing is applied
• Results are deterministic
"""
        wx.MessageBox(help_text, "Help", wx.OK | wx.ICON_INFORMATION)

    def on_analyze(self, event):
        if not self.image_path:
            wx.MessageBox("Please open an image first", "Error", wx.ICON_ERROR)
            return

        try:
            rgb = dominant_color_histogram(
                self.image_path,
                bins_per_channel=self.bin_ctrl.GetValue(),
                blur=self.blur_ctrl.GetValue()
            )
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.ICON_ERROR)
            return

        self.result_text.SetLabel(f"Dominant RGB: {rgb}")
        self.color_panel.SetBackgroundColour(wx.Colour(*rgb))
        self.color_panel.Refresh()


class DominantColorApp(wx.App):
    def OnInit(self):
        frame = DominantColorFrame()
        frame.Show()
        return True


if __name__ == "__main__":
    app = DominantColorApp(False)
    app.MainLoop()

