#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Perceptual/RegularisedSynthesis/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                agent { label "gpu" }
                axes {
                    axis {
                        name 'synth'
                        values 'freq_harmonic', 'full_harmonic', 'inharmonic',
                    }
                    axis {
                        name 'detnoise'
                        values '', 'detnoise_'
                    }
                }
                stages {
                    steps {
                        stage("Run experiment") {
                            script {
                                echo "+=+=+=+=+=====> Running experiment: ${detnoise}${synth}"
                                def exp = "${detnoise}${synth}"
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
        stage("Run STFT experiment") {
            agent { label "gpu" }
            script {
                sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name stft \
                        -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE} \
                        python3 ${EXP_DIR}/attacks.py stft --max_spawns 5
                """
            }
        }
    }
}
