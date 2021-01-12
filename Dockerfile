FROM dijksterhuis/cleverspeech:build


RUN mkdir -p experiments/Baselines/
RUN mkdir -p experiments/AdditiveSynthesis/
RUN mkdir -p experiments/CTCHiScores/
RUN mkdir -p experiments/SimpleHiScores/

COPY ./Baselines/ ./experiments/Baselines/
COPY ./AdditiveSynthesis/ ./experiments/AdditiveSynthesis/
COPY ./CTCHiScores/ ./experiments/CTCHiScores/
COPY ./SimpleHiScores/ ./experiments/SimpleHiScores/


