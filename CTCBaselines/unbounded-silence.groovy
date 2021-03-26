#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    options { skipDefaultCheckout() }
    parameters {
            string(name: 'MAX_SPAWNS', defaultValue: '5', description: 'Number of attacks to spawn at once.')
            string(name: 'N_STEPS', defaultValue: '10000', description: '')
    }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/CTCBaselines/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                agent { label "gpu" }
                axes {
                    axis {
                        name 'experiment'
                        values 'ctc', 'ctc2'
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
                                echo "+=+=+=+=+=====> Running experiment: ${experiment}"
                                def exp = "${experiment}"
                                sh """
                                    docker run \
                                        --gpus device=${GPU_N} \
                                        -t \
                                        --rm \
                                        --shm-size=10g \
                                        --pid=host \
                                        --name ${exp} \
                                        -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                                        -e LOCAL_UID=\$(id -u ${USER}) \
                                        -e LOCAL_GID=\$(id -g ${USER}) \
                                        ${IMAGE} \
                                        python3 ${EXP_DIR}/unbounded.py ${exp} \
                                            --max_spawns "${params.MAX_SPAWNS}" \
                                            --nsteps "${params.N_STEPS}" \
                                            --audio_indir "./silence/wav/" \
                                            --targets_path "./silence/cv-valid-test.csv"
                                """
                                sh "tar -cvz -f ${exp}_\$(date +%y%m%d_%H%M%S).tar.gz ./results/"
                            }
                        }
                    }
                }
                post {
                    always {
                        sh "docker container prune -f"
                        sh "docker image prune -f"
                    }
                }
            }
        }
    }
}
