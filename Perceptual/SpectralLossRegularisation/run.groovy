#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Perceptual/SpectralLossRegularisation"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments."){
            failFast false
            matrix {
                agent { label "gpu" }
                axes {
                    axis {
                        name 'experiment'
                        values 'multi_scale', 'spectral'
                    }
                }
                stages {
                    steps {
                        stage("Image pull") {
                            script {
                                sh "docker pull ${IMAGE}"
                            }
                        }
                        stage("Run experiment") {
                            script {
                                echo "+=+=+=+=+=====> Running experiment: ${experiment}"
                                def exp = "${experiment}"
                                sh """
                                    docker run \
                                        --gpus device=${GPU_N} \
                                        -t \
                                        --rm \
                                        --name ${exp} \
                                        -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                                        -e LOCAL_UID=\$(id -u ${USER}) \
                                        -e LOCAL_GID=\$(id -g ${USER}) \
                                        ${IMAGE} \
                                        python3 ${EXP_DIR}/attacks.py ${exp} --max_spawns 5
                                """
                            }
                        }
                    }
                }
            }
        }
    }
}