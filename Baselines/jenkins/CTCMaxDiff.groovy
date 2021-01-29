#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE_NAME = "dijksterhuis/cleverspeech"
        TAG = "latest"
        BEAM_EXP_ARG = "ctcmaxdiff_beam"
        GREEDY_EXP_ARG = "ctcmaxdiff_greedy"
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
                        ${GREEDY_EXP_ARG} \
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
                        ${BEAM_EXP_ARG} \
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