import collections
import json
import os

import datasets


_HOMEPAGE = "https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/1?ref=roboflow2huggingface"
_LICENSE = "CC BY 4.0"
_CITATION = """\
@misc{ vehicle-registration-plates-trudk_dataset,
    title = { Vehicle Registration Plates Dataset },
    type = { Open Source Dataset },
    author = { Augmented Startups },
    howpublished = { \\url{ https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk } },
    url = { https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { jun },
    note = { visited on 2023-01-18 },
}
"""
_CATEGORIES = ['license_plate']
_ANNOTATION_FILENAME = "_annotations.coco.json"


class LICENSEPLATEOBJECTDETECTIONConfig(datasets.BuilderConfig):
    """Builder Config for license-plate-object-detection"""

    def __init__(self, data_urls, **kwargs):
        """
        BuilderConfig for license-plate-object-detection.

        Args:
          data_urls: `dict`, name to url to download the zip file from.
          **kwargs: keyword arguments forwarded to super.
        """
        super(LICENSEPLATEOBJECTDETECTIONConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_urls = data_urls


class LICENSEPLATEOBJECTDETECTION(datasets.GeneratorBasedBuilder):
    """license-plate-object-detection object detection dataset"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        LICENSEPLATEOBJECTDETECTIONConfig(
            name="full",
            description="Full version of license-plate-object-detection dataset.",
            data_urls={
                "train": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/train.zip",
                "validation": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/valid.zip",
                "test": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/test.zip",
            },
        ),
        LICENSEPLATEOBJECTDETECTIONConfig(
            name="mini",
            description="Mini version of license-plate-object-detection dataset.",
            data_urls={
                "train": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/valid-mini.zip",
                "validation": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/valid-mini.zip",
                "test": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/valid-mini.zip",
            },
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    {
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=_CATEGORIES),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(self.config.data_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "folder_dir": data_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "folder_dir": data_files["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "folder_dir": data_files["test"],
                },
            ),
]

    def _generate_examples(self, folder_dir):
        def process_annot(annot, category_id_to_category):
            return {
                "id": annot["id"],
                "area": annot["area"],
                "bbox": annot["bbox"],
                "category": category_id_to_category[annot["category_id"]],
            }

        image_id_to_image = {}
        idx = 0

        annotation_filepath = os.path.join(folder_dir, _ANNOTATION_FILENAME)
        with open(annotation_filepath, "r") as f:
            annotations = json.load(f)
        category_id_to_category = {category["id"]: category["name"] for category in annotations["categories"]}
        image_id_to_annotations = collections.defaultdict(list)
        for annot in annotations["annotations"]:
            image_id_to_annotations[annot["image_id"]].append(annot)
        filename_to_image = {image["file_name"]: image for image in annotations["images"]}

        for filename in os.listdir(folder_dir):
            filepath = os.path.join(folder_dir, filename)
            if filename in filename_to_image:
                image = filename_to_image[filename]
                objects = [
                    process_annot(annot, category_id_to_category) for annot in image_id_to_annotations[image["id"]]
                ]
                with open(filepath, "rb") as f:
                    image_bytes = f.read()
                yield idx, {
                    "image_id": image["id"],
                    "image": {"path": filepath, "bytes": image_bytes},
                    "width": image["width"],
                    "height": image["height"],
                    "objects": objects,
                }
                idx += 1
