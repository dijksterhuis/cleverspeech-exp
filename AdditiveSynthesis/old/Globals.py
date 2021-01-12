TARGETS = [
        "open the door",
        "your cat sat on the big rug",
        "i'm glad she was able to do all the work",
        "why did it take so long for him to get into the system",
    ]

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
NUMB_STEPS = 5000
DECODING_STEP = 10
BEAM_WIDTH = 500

INDIR = "./samples"
OUTDIR = "./adv/addsynth/"

RESCALE = 0.95

N_OSC = [32, 16, 8, 4, 2]

#DISTANCE_LOSS_WEIGHTING = [1.0, 0.1, 0.01, 0.001]
#EXP_EPSILONS = [i for i in range(11, 4, -3)]
