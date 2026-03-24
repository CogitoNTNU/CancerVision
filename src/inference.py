import torch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, EnsureTyped
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet  # Eller den arkitekturen du brukte

# 1. Definer hvordan bildet skal leses og klargjøres
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,  # F.eks. Bakgrunn vs. Svulst/Vev
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

model.load_state_dict(torch.load("din_modell_vekter.pth"))
model.eval()  # Setter modellen i "inference-modus"

with torch.no_grad():  # Sparer minne ved å ikke beregne gradienter
    # Last inn et bilde (f.eks. en .nii.gz fil)
    data = test_transforms({"image": "hjerne_scan_01.nii.gz"})
    input_tensor = data["image"].unsqueeze(0).to(device)

    # Kjør segmentering med sliding window
    output = sliding_window_inference(
        inputs=input_tensor,
        roi_size=(96, 96, 96),  # Størrelsen på "ruta" som scanner
        sw_batch_size=4,
        predictor=model,
    )

    # Gjør om sannsynligheter til faktiske merkelapper (0 eller 1)
    segmentation_mask = torch.argmax(output, dim=1).detach().cpu().numpy()
