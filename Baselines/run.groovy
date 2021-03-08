#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE_NAME = "dijksterhuis/cleverspeech"
        TAG = "latest"
        EXP_DIR = "./experiments/Baselines"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }

    stages {

        stage('Prep work.') {
            steps {
                script {
                    sh "docker container prune -f"
                    sh "docker pull ${IMAGE_NAME}:${TAG}"
                }
            }
        }
        stage("Run baseline CTC experiment."){
            steps {
                script {

                    sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name ${EXP_ARG} \
                        -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE_NAME}:${TAG} \
                        python3 \
                        ${EXP_DIR}/attacks.py \
                        ctc \
                        --max_spawns 5
                    """
                }
            }
        }
        stage("Run CTC V2 experiment."){
            steps {
                script {

                    sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name ${EXP_ARG} \
                        -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE_NAME}:${TAG} \
                        python3 \
                        ${EXP_DIR}/attacks.py \
                        ctc_v2 \
                        --max_spawns 5
                    """
                }
            }
        }
        stage("Run CW Max Diff attack with CTC alignment search and greedy search decoder."){
            steps {
                script {
                    sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name ${GREEDY_EXP_ARG} \
                        -v \$(pwd)/results/greedy/:${CLEVERSPEECH_HOME}/adv \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE_NAME}:${TAG} \
                        python3 \
                        ${EXP_DIR}/attacks.py \
                        ctcalign_maxdiff_greedy \
                        --max_spawns 5
                    """
                }
            }
        }

        stage("Run CW Max Diff attack with CTC alignment search and DS beam search decoder."){
            steps {
                script {
                    sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name ${BEAM_EXP_ARG} \
                        -v \$(pwd)/results/beam/:${CLEVERSPEECH_HOME}/adv \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE_NAME}:${TAG} \
                        python3 \
                        ${EXP_DIR}/attacks.py \
                        ctcalign_maxdiff_beam \
                        --max_spawns 5
                    """
                }
            }
        }
    }
    post  {
        always {
            sh "docker image rm ${IMAGE_NAME}:${TAG}"
            sh "docker image prune -f"
            sh "docker container prune -f"
        }
    }
}