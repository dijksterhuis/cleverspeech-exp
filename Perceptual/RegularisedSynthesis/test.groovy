#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    parameters {
            string(name: 'MAX_SPAWNS', defaultValue: '2', description: '')
            string(name: 'N_STEPS', defaultValue: '10', description: '')
            string(name: 'DECODE_STEP', defaultValue: '2', description: '')
            string(name: 'BATCH_SIZE', defaultValue: '2', description: '')
            string(name: 'MAX_EXAMPLES', defaultValue: '4', description: '')
    }
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
                        values 'freq_harmonic', 'full_harmonic', 'inharmonic'
                    }
                    axis {
                        name 'detnoise'
                        values '', 'detnoise_'
                    }
                }
                stages {
                    stage("Image pull") {
                        steps {
                            script {
                                sh "docker pull ${IMAGE}"
                            }
                        }
                    }
                    stage("Run experiment") {
                        steps {
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
                                        python3 ${EXP_DIR}/attacks.py ${exp} \
                                            --max_spawns "${params.MAX_SPAWNS}" \
                                            --batch_size "${params.BATCH_SIZE}" \
                                            --decode_step "${params.DECODE_STEP}" \
                                            --nsteps "${params.N_STEPS}" \
                                            --max_examples "${params.MAX_EXAMPLES}"
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
                        python3 ${EXP_DIR}/attacks.py stft \
                            --max_spawns "${params.MAX_SPAWNS}" \
                            --batch_size "${params.BATCH_SIZE}" \
                            --decode_step "${params.DECODE_STEP}" \
                            --nsteps "${params.N_STEPS}" \
                            --max_examples "${params.MAX_EXAMPLES}"
                """
            }
        }
    }
}
