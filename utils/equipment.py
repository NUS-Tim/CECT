import torch


def equipment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Available cuda detected:", torch.cuda.get_device_name(0))
    else:
        print("Cuda unavailable, switch to %s instead" % device)

    return device
