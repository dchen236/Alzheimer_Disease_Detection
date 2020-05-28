from torchvision import transforms

RES18 = "res18"
RES34 = "res34"
RES50 = "res50"
RES101 = "res101"
RES152 = "res152"
SQZ = "squeeze"
VGG = "vgg"
ALX = "alexnet"
DENSE = "densenet"
input_size = 244
MODELS_SETTINGS = { RES18:{"batch": 128,
                           "epoch":50},
                    RES34:{"batch": 128,
                           "epoch":50},
                    RES50:{"batch": 32,
                           "epoch":50},
                   RES101:{"batch": 32,
                           "epoch":50},
                   RES152:{"batch": 20,
                           "epoch":50},
                      SQZ:{"batch": 128,
                           "epoch":50},
                      VGG:{"batch": 32,
                           "epoch":50},
                      ALX:{"batch": 32,
                           "epoch":50},
                    DENSE:{"batch": 32,
                           "epoch":50}}
MODELS = [RES18, RES34, RES50, RES101, RES152, VGG, SQZ, ALX, DENSE]
MAX_EPs = [MODELS_SETTINGS[model]['epoch'] for model in MODELS]
BATCH_SIZE = [MODELS_SETTINGS[model]['batch'] for model in MODELS]
SAVE_HIST = ["train_histories/" + model + "_hist.json" for model in MODELS]
SAVE_OPT =  ["optimizers/" + model + "_optimizer.pt" for model in MODELS]
SAVE_MODELS =  ["models/" + model + "_best.pt" for model in MODELS]

data_transforms = {"train": transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((input_size, input_size)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485,), (0.229,))]),
                            "val": transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((input_size, input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,), (0.229,))])}

label_dict = {0:"VeryMildDemented",
              1:"ModerateDemented",
             2:"MildDemented",
              3:"NonDemented"}