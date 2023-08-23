import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.cuda.nvtx as nvtx
import argparse
from pynvml.smi import nvidia_smi

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--bs-train", help="batch size for training", type=int, default=512)
parser.add_argument("--bs-valid", help="batch size for validation", type=int, default=512)
parser.add_argument("--half", help="use half-precision floating-point", action='store_true')
parser.add_argument("-N", "--num-iterations", help="the number of iterations", type=int, default=128)
parser.add_argument("--power-samples", help="the number of power samples per experiment", type=int, default=100)
args = parser.parse_args()
print(args)

# General parameters
device = 'cuda'
W = 224
H = 224
D = 3
C = 1000


def testInference(model, output):
    """
    Tests model inference on the given model
    :param model:
    :param output:
    :return:
    """

    # Generate random input
    x = torch.rand(args.bs_valid, D, W, H, device=device, dtype=torch.float16 if args.half else torch.float32)
    torch.cuda.synchronize()

    nvtx.range_push(f'Testing inference with {model.__class__.__name__}')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    totalPower = 0
    numMeasurements = 0

    model.eval()
    with torch.no_grad():
        for i in range(args.num_iterations):
            model(x)
            
            if i % (args.num_iterations // args.power_samples) == 0:
                totalPower += nvidia_smi.getInstance().DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw']
                numMeasurements += 1
                torch.cuda.synchronize()

    end.record()
    torch.cuda.synchronize()

    nvtx.range_pop()

    output.write(f'Testing inference with {model.__class__.__name__}: {round(totalPower / numMeasurements, 0)}W and {round(start.elapsed_time(end) / 1000, 2)}s\n')
    
    
def testTraining(model, output):
    """
    Tests model training on the given model
    :param model:
    :param output:
    :return:
    """

    # Generate random input
    x = torch.rand(args.bs_train, D, W, H, device=device)
    y = torch.randint(0, C, size=(args.bs_train, ), device=device)
    torch.cuda.synchronize()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    nvtx.range_push(f'Testing training with {model.__class__.__name__}')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    totalPower = 0
    numMeasurements = 0

    model.train()
    for i in range(args.num_iterations):

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if i % (args.num_iterations // args.power_samples) == 0:
            totalPower += nvidia_smi.getInstance().DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw']
            numMeasurements += 1
            torch.cuda.synchronize()

    end.record()
    torch.cuda.synchronize()

    nvtx.range_pop()

    output.write(f'Testing training with {model.__class__.__name__}: {round(totalPower / numMeasurements, 0)}W and {round(start.elapsed_time(end) / 1000, 2)}s\n')


if __name__ == '__main__':

    output = open("output", "w")
    
    torch.backends.cudnn.allow_tf32 = False
    
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device=device)
    if args.half:
        model = model.half()
    testInference(model, output)  # 724705896 MACs
    if not args.half:
        testTraining(model, output)  # 3332131161 MACs
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device=device)
    if args.half:
        model = model.half()
    testInference(model, output)  # 4297355192 MACs
    if not args.half:
        testTraining(model, output)  # 12822701741 MACs
        
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT).to(device=device)
    if args.half:
        model = model.half()
    testInference(model, output)  # 1255899984 MACs
    if not args.half:
        testTraining(model, output)  # 4042532867 MACs

    output.close()
