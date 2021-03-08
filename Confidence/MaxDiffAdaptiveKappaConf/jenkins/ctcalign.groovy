#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE_NAME = "dijksterhuis/cleverspeech"
        TAG = "latest"
        GITHUB_BRANCH = "master"
        EXP_DIR = "./experiments/Confidence/MaxDiffAdaptiveKappaConf/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Prep work.") {
            steps {
                script {
                    withDockerRegistry([ credentialsId: "dhub-mr", url: "" ]) {
                        sh "docker container prune -f"
                        sh "docker pull ${IMAGE_NAME}:${TAG}"
                    }
                }
            }
        }

        stage("Run dense only experiment."){
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
                        ctcalign \
                        --max_spawns 5
                    """
                }
            }
        }
        stage("Run dense rctc experiment."){
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
                        ctcalign-rctc \
                        --max_spawns 5
                    """
                }
            }
        }
        stage("Run dense ctc experiment."){
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
                        ctcalign-cctc \
                        --max_spawns 5
                    """
                }
            }
        }
    }
    post  {
        always {
            sh "docker image prune -f"
            sh "docker container prune -f"
            sh "docker image rm ${IMAGE_NAME}:${TAG}"
        }
    }
}