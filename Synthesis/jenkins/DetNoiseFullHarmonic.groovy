#!/usr/bin/env groovy

pipeline {
    agent { label "kalluto" }
    environment {
        IMAGE_NAME = "dijksterhuis/cleverspeech"
        TAG = "latest"
        GITHUB_BRANCH = "master"
        EXP_ARG = "detnoise_full_harmonic"
        EXP_DIR = "./experiments/Synthesis"
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

        stage("Run experiment."){
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
                        ${EXP_ARG} \
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