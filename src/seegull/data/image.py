"""Utilities for working with images.

This module contains utilities for working with images. The primary export is
the `Image` class which has functions for manipulating images and using
supported models to detect objects in images. There are also
helper functions for loading images from Pandas DataFrames.
"""

import base64
import hashlib
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
import pandas as pd
import PIL
import requests
import supervision as sv
from google.cloud import storage
from pillow_heif import register_heif_opener
from tqdm.contrib.concurrent import process_map
from typing_extensions import Self

# Enable reading heic files
register_heif_opener()

# from bower_ml.models.autodistill import AutodistillModelWrapper
# from bower_ml.models.yolo import YOLO, YOLOMultiLabel


class Image:
    """A class for (down)loading, mainpulating and predicting on images."""

    def __init__(
        self,
        path: Path | str = None,
        image: np.ndarray = None,
        url: str = None,
        low_memory: bool = False,
        force_redownload: bool = False,
        file_extension: str = None,
        **kwargs,
    ):
        """
        Args:
            path: The local path to either where the file is stored or where
                it should be downloaded to. If None, a temporary file will be
                used instead.
            image: The image pixels as a uint8 numpy array with shape
                [height, width, 3]. The array should be in BGR format.
            url: A remote path to the file. If provided, the image will be
                downloaded during initialization of `Image`.
                Supported protocals are:
                    http(s)://, gs:// (Google Storage)
            low_memory: If this is True the raw pixels won't be kept in
                memory. Otherwise the pixels will be stored at `self.image`
                after the first load.
            force_redownload: If True, redownload the image from `url` even
                even if it already exists at `path`.
            file_extension: If either path or url is given, file_extension
                will be overwritten by the inferred value. This is useful when
                Image is initialized from `image` pixels.

        Raises:
            ValueError: If none of `path`, `url` or `image` are provided.
        """
        self._path = Path(path) if path else path
        self._image = image
        self.url = url
        self.low_memory = low_memory

        # Try to infer the file extension
        self.file_extension = file_extension
        if self._path:
            self.file_extension = self._path.suffix.removeprefix(".")
        elif self.url:
            filename = self.url.split("/")[-1]
            self.file_extension = filename.split(".")[-1]

        if (
            (not self.path_exists())
            and self._image is None
            and self.url is None
        ):
            raise ValueError("One of path, image or url must be set")

        self.download(force=force_redownload)

    @property
    def image(self):
        # Load the image if it's not already loaded and the path exists
        if self._image is None and self.path_exists():
            image = cv2.cvtColor(
                np.array(PIL.Image.open(self.path).convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )

            # Only save it in memory if low_memory isn't True.
            # Otherwise it will be reloaded from disk each time
            # `.image` is accessed.
            if not self.low_memory:
                self._image = image

            return image

        return self._image

    @property
    def path(self):
        # If the path doesn't exist but the image is set and the
        # `path` property is being accessed, store the image in a tmp file
        # so that the accessor can access the image from a path on disk.
        if (not self.path_exists()) and (self._image is not None):
            self.mktmp()
            self.save(self._path)

        return self._path

    def path_exists(self):
        return self._path and self._path.exists()

    def mktmp(self):
        if not self.path_exists():
            filename = hashlib.sha256(
                f"{self.url} {self._image}".encode()
            ).hexdigest()
            self._path = Path(f"/tmp/{filename}.{self.file_extension}")

    @property
    def rgb(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def shape(self):
        return self.image.shape[:2]

    def download(self, force=False):
        if not force and (not self.url or self.path_exists()):
            return

        if self._path is None:
            self.mktmp()

        if self.url.startswith("http"):
            resp = requests.get(self.url, stream=True)
            resp.raise_for_status()
            with self.path.open("wb") as f:
                for chunk in resp:
                    f.write(chunk)
        elif self.url.startswith("gs"):
            storage_client = storage.Client()
            with self.path.open("wb") as f:
                storage_client.download_blob_to_file(self.url, f)
        else:
            raise ValueError(f"Unsupported Protocol: {self.url}")

    # def predict(
    #     self,
    #     model: YOLO | DetectionBaseModel | AutodistillModelWrapper,
    #     conf=0.5,
    #     nms_threshold=0.5,
    # ) -> Self:
    #     if isinstance(model, YOLOMultiLabel):
    #         classes = model.idx_to_name
    #         self.yolo_result = model(self.path, conf=conf, verbose=False)[0]
    #         class_id = [
    #             tuple(c)
    #             for c in self.yolo_result.boxes.cls.cpu().numpy().astype(int)
    #         ]
    #         self.sv_detection = sv.Detections(
    #             xyxy=self.yolo_result.boxes.xyxy.cpu().numpy(),
    #             confidence=self.yolo_result.boxes.conf.cpu()
    #             .numpy()
    #             .prod(axis=1),
    #             class_id=np.array([model.tuple_to_idx[t] for t in class_id]),
    #         )
    #     elif isinstance(model, YOLO):
    #         classes = model.classes
    #         self.yolo_result = model(self.path, conf=conf, verbose=False)[0]
    #         self.sv_detection = sv.Detections.from_ultralytics(self.yolo_result)
    #     elif isinstance(model, DetectionBaseModel):
    #         classes = model.ontology.classes()
    #         self.sv_detection = model.predict(self.image)
    #     elif isinstance(model, AutodistillModelWrapper):
    #         classes = model.classes
    #         self.sv_detection = model.predict(self, conf=conf)
    #     else:
    #         raise NotImplementedError(f"Model {model} is not supported.")

    #     self.sv_detection = self.sv_detection.with_nms(
    #         threshold=nms_threshold, class_agnostic=True
    #     )

    #     self.labels = [
    #         f"{classes[class_id]} {confidence:0.2f}"
    #         for _, _, confidence, class_id, _, _ in self.sv_detection
    #     ]

    #     return self

    # def annotate(self) -> Self:
    #     box_annotator = sv.BoxAnnotator()

    #     annotated_image = box_annotator.annotate(
    #         scene=self.image.copy(),
    #         detections=self.sv_detection,
    #         labels=self.labels,
    #     )

    #     im = Image(
    #         image=annotated_image,
    #         file_extension=self.file_extension,
    #     )
    #     im.sv_detection = self.sv_detection
    #     im.labels = self.labels
    #     return im

    # def annotate_row(
    #     self, rows: pd.Series | pd.DataFrame, labels: str | list[str]
    # ) -> Self:
    #     """
    #     Annotate based on a Series or DataFrame with bounding boxes in
    #     x1, y1, x2, y2 format.
    #     """
    #     if isinstance(rows, pd.Series):
    #         rows = pd.DataFrame([rows])
    #     if isinstance(labels, str):
    #         labels = [labels]

    #     # Get bounding box
    #     xyxy = rows[["x1", "y1", "x2", "y2"]].values

    #     # Scale annotations if rows have image_width and image_height
    #     if ("image_width" in rows.columns) and ("image_height" in rows.columns):
    #         sw = self.width / rows.image_width
    #         sh = self.height / rows.image_height
    #         scale = np.dstack([sw, sh, sw, sh])[0]
    #         xyxy = xyxy * scale

    #     self.sv_detection = sv.Detections(xyxy=xyxy)
    #     self.labels = labels
    #     return self.annotate()

    def crop(
        self,
        xyxy: Sequence[int],  # | None = None,
        padding: int | float = 0,
        **kwargs,
    ) -> Self:
        """Crop the image to the given bounding box, optionally with padding.

        Args:
            xyxy: The bounding box in xyxy format
            padding: Optionally pad the bounding box by a given number of
            pixels (int) or by a percentage of the bounding box (float)

        Returns:
            The cropped image as a new `Image`.
        """
        # if xyxy is None:
        #    xyxy = self.sv_detection.xyxy[0].astype(int)

        x1, y1, x2, y2 = xyxy

        if padding:
            if isinstance(padding, float):
                px = int(padding * (x2 - x1))
                py = int(padding * (y2 - y1))
            else:
                px = py = padding

            x1 = max(0, x1 - px)
            x2 += px
            y1 = max(0, y1 - py)
            y2 += py

        return Image(
            image=self.image[y1:y2, x1:x2],
            file_extension=self.file_extension,
        )

    # def detection_crops(
    #     self, padding: int | float = 0
    # ) -> Sequence[tuple[Sequence[int], float, Self]]:
    #     """
    #     Return a list of tuples, one for each object in
    #     self.sv_detection.

    #     Optionally pad each cropped image. This won't change the returned
    #     xyxy values to reflect the changed crop.

    #     Each tuple contains (xyxy, conf, class_id, cropped image).
    #     """
    #     return [
    #         (
    #             xyxy,
    #             self.sv_detection.confidence[i],
    #             self.sv_detection.class_id[i],
    #             self.crop(xyxy, padding),
    #         )
    #         for i, xyxy in enumerate(self.sv_detection.xyxy.astype(int))
    #     ]

    def resize(
        self,
        w: int | float,
        h: int | None = None,
        min_width: int | None = None,
        min_height: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> Self:
        """Resize the image to the given width and height.

        Accepts either integers, in which case they're interpreted as
        pixel values, or a single between (0, 1] in which case it's
        interpreted as a percentage. The float is used for both dimensions.

        When using percentage inputs, min_width, min_height can be used
        to prevent the image from getting smaller than those values.

        Args:
            w: Either an integer or float. If an integer, the number of pixels
                to resize the width to. If it's a float, it's the percentage to
                resize to and is used for both dimensions.
            h: If `w` is an integer, optionally `h` can also be given as an
                integer. If `w` is an integer and `h` is None, then `w` will be
                used for both dimensions.
            min_width, min_height, max_width, max_height: Optional constraints
                that can be provided when resizing using a percentage. The
                image won't be made smaller or larger in any dimension than
                the minimums and maximums and it's aspect ratio will be
                maintained.

        Returns:
            The resized image as a new `Image`.
        """
        # Handle percentages
        if isinstance(w, float):
            h = w

            # Handle min_width and min_height
            if min_width is not None:
                w = max(min_width, self.width * w) / self.width
            if min_height is not None:
                h = max(min_height, self.height * h) / self.height

            w = h = max(w, h)

            # Handle max_width and max_height
            if max_width is not None:
                w = min(max_width, self.width * w) / self.width
            if max_height is not None:
                h = min(max_height, self.height * h) / self.height

            p = min(w, h)

            # Convert to integers
            w = int(self.width * p)
            h = int(self.height * p)

        if h is None:
            h = w

        image = cv2.resize(
            self.image, dsize=(w, h), interpolation=cv2.INTER_CUBIC
        )
        return Image(
            image=image,
            file_extension=self.file_extension,
        )

    def pad(self, padding: int, color=[255, 255, 255]):
        """Pad the image with a constant color.

        Args:
            padding: The number of pixels to pad on all sides.

        Returns:
            The padded image as a new `Image`.
        """
        return Image(
            image=cv2.copyMakeBorder(
                self.image,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=color,
            ),
            file_extension=self.file_extension,
        )

    def display(self, *args, **kwargs) -> None:
        """Plot the image using matplotlib.

        This will display the image in a Jupyter notebook or Colab.

        See https://supervision.roboflow.com/utils/notebook/#plot_image for
        details on accepted parameters.
        """
        sv.plot_image(self.image, *args, **kwargs)

    def image_bytes(self) -> bytes:
        """Return the image as jpg encoded bytes."""
        return cv2.imencode(".jpg", self.image)[1].tobytes()

    def base64(self) -> bytes:
        """Return the image as a base64 encoded jpg bytes"""
        return base64.b64encode(self.image_bytes())

    def to_pil(self) -> PIL.Image:
        """Convert the image to a PIL.Image instance"""
        return PIL.Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def save(self, path: Path):
        """Save the image to disk at its path."""
        self.to_pil().save(path)

    def __hash__(self) -> int:
        return hash(self.image_bytes())


def load_image(
    row: pd.Series,
    return_type: Literal["seegull", "PIL"] = "seegull",
    resize: float | tuple[int, int] | None = None,
    crop: bool = False,
    image_source_column_name: str = "image_source",
    path_column_name: str = "path",
    **kwargs,
):
    """Loads an image from a row of a seegull formatted DataFrame.

    The DataFrame must have at least one of the following:
        - A column with the url to the image (default column: "image_source")
        - A column with the path to the image (default column: "path")

    If both path and url are provided and the image does not exist on disk
    at the path, it will be downloaded to the given path. If it does not
    exist and no path is given it will be downloaded to a temporary file.

    If crop is True, then the row must have attributes `x1`, `y1`, `x2`, `y2`
    defining the bounding box to crop.

    See the `__init__`, `crop` and `resize` methods of `Image` for additional
    kwargs that can be passed.

    Args:
        row: A row of a Dataframe with the attributes described above
        return_type: Whether to return a `seegull.Image` or a `PIL.Image`
        resize: A float or (w, h) pair to resize the image
        crop: Whether to crop the image to the given bounding box
        image_source_column_name: The column to use as the image_source/url
        path_column_name: The column to use as the path

    Returns:
        Either a `seegull.Image` or `PIL.Image`

    Raises:
        NotImplementedError: If an unknown return_type is passed
    """
    im = Image(
        url=getattr(row, image_source_column_name, None),
        path=getattr(row, path_column_name, None),
        **kwargs,
    )

    if crop:
        im = im.crop(
            np.array([row.x1, row.y1, row.x2, row.y2]).astype(int), **kwargs
        )

    if resize:
        if isinstance(resize, float):
            im = im.resize(resize, **kwargs)
        else:
            im = im.resize(*resize, **kwargs)

    if return_type == "bower_ml":
        return im
    elif return_type == "PIL":
        return im.to_pil()
    else:
        raise NotImplementedError(f"Unknown return type: {return_type}")


def load_images(
    df: pd.DataFrame,
    # return_type: Literal["bower_ml", "PIL"] = "bower_ml",
    # resize: tuple[int, int] | None = None,
    max_workers: int | None = None,
    # low_memory: bool = True,
    # crop: bool = False,
    # force_redownload: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load the images in the given DataFrame.

    See `load_image` above for a description of the expected DataFrame format
    and possible kwargs.

    Args:
        df: The DataFrame to load images from
        max_workers: The number of concurrent workers to use.
            See https://tqdm.github.io/docs/contrib.concurrent/#process_map
    """

    # if crop:
    #     assert len(set(["x1", "y1", "x2", "y2"]) & set(df.columns)) == 4

    images = process_map(
        partial(
            load_image,
            # return_type=return_type,
            # resize=resize,
            # low_memory=low_memory,
            # crop=crop,
            # force_redownload=force_redownload,
            **kwargs,
        ),
        [row for _, row in df.iterrows()],
        desc="loading images",
        max_workers=max_workers,
    )

    return pd.Series(images)


def display_image_df(df: pd.DataFrame, print_cols=None):
    """Display the images in a DataFrame."""
    for i, row in df.iterrows():
        if print_cols:
            print(*[row[col] for col in print_cols])
        load_image(row).display()
