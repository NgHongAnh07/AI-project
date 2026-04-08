import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
)


def build_model(backbone, num_classes):
    # Ensure at least 2 classes (background + object)
    if num_classes == 1:
        num_classes = 2
        print("--> Auto-adjusting num_classes to 2 (Background + Laptop)")

    # Choose backbone
    if "mobilenet" in backbone.lower():
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights
        )
    else:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights
        )

    # Get number of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace head with new predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model