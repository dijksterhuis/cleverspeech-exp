#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    options { skipDefaultCheckout() }
    parameters {
        string(name: 'MAX_SPAWNS', defaultValue: '2', description: '')
            string(name: 'N_STEPS', defaultValue: '100', description: '')
            string(name: 'DECODE_STEP', defaultValue: '10', description: '')
            string(name: 'SPAWN_DELAY', defaultValue: '5', description: '')
        string(name: 'BATCH_SIZE', defaultValue: '2', description: '')
        string(name: 'MAX_EXAMPLES', defaultValue: '4', description: '')
    }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Confidence/CWMaxDiffBaselines/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                agent { label "gpu" }
                axes {
                    axis {
                        name 'alignment'
                        values 'sparse', 'dense', 'ctcalign'
                    }
                    axis {
                        name 'decoder'
                        values 'beam', 'greedy'
                    }
                }
                stages {
                    stage("Locked SCM checkout") {
                        steps {
                            lock("dummy") {
                                sleep 5
                                checkout scm
                            }
                        }
                    }
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
                                echo "+=+=+=+=+=====> Running experiment: ${alignment}-${decoder}"
                                def exp = "${alignment}-${decoder}"
                                sh """
                                    docker run \
                                        --gpus device=${GPU_N} \
                                        -t \
                                        --rm \
                                        --shm-size=10g \
                                        --name ${exp} \
                                        -e LOCAL_UID=9999 \
                                        -e LOCAL_GID=9999 \
                                        ${IMAGE} \
                                        python3 ${EXP_DIR}/attacks.py ${exp} \
                                            --max_spawns "${params.MAX_SPAWNS}" \
                                            --spawn_delay "${params.SPAWN_DELAY}" \
                                            --batch_size "${params.BATCH_SIZE}" \
                                            --decode_step "${params.DECODE_STEP}" \
                                            --nsteps "${params.N_STEPS}" \
                                            --max_examples "${params.MAX_EXAMPLES}"
                                """
                            }
                        }
                    }
                }
                /* post {
                    always {
                        sh "docker image prune -f"
                        sh "docker container prune -f"
                        sh "docker image rm ${IMAGE}"
                    }
                } */
            }
        }
    }
}
