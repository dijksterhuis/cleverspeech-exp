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
        EXP_DIR = "./experiments/Confidence/AdaptiveKappa/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                agent { label "gpu" }
                axes {
                    axis {
                        name 'alignment_type'
                        values 'dense', 'sparse', 'ctcalign'
                    }
                    axis {
                        name 'loss_type'
                        values 'none', 'rctc', 'ctc'
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
                                echo "+=+=+=+=+=====> Running experiment: ${alignment_type}-${loss_type}"
                                def exp = "${alignment_type}-${loss_type}"
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
    }
}


