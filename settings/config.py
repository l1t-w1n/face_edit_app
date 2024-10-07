import torch


class Generation:
    default_psi = 0.75
    model_path = "models/ffhq.pkl"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")


print(torch.version.cuda)  # Should print 12.x if PyTorch is built with CUDA 12.x
print(torch.cuda.is_available())  # Should print True if CUDA is available
print(torch.cuda.get_device_name(0))  # Should print your GPU's name


class Projection:
    generation_dir = "generated/"
    alignment_dir = "input_imgs/"
    alignment_model = 'models/shape_predictor_68_face_landmarks.dat'
    num_steps = 100
    save_video = True
    seed = 305


class Shifting:
    vectors_path = "vectors/"
    extension = ".npy"


class GUIConfig:
    theme_name = "DarkTeal9"
    display_size = (400, 400)
    shift_range = (-10, 10)
    vector_names = (
        "age",
        "eye_distance",
        "eye_eyebrow_distance",
        "eye_ratio",
        "eyes_open",
        "gender",
        "lip_ratio",
        "mouth_open",
        "mouth_ratio",
        "nose_mouth_distance",
        "nose_ratio",
        "nose_tip",
        "pitch",
        "roll",
        "smile",
        "yaw",
    )


class Config:
    generation = Generation
    shifting = Shifting
    gui = GUIConfig
    projection = Projection