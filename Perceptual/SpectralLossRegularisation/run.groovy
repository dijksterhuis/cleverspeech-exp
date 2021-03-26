#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    options { skipDefaultCheckout() }
    parameters {
            string(name: 'MAX_SPAWNS', defaultValue: '3', description: 'Number of attacks to spawn at once.')
        }
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
                        values 'multiscale-L1', 'multiscale-L2', 'spectral'
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
                                    python3 ${EXP_DIR}/attacks.py ${exp} --max_spawns "${params.MAX_SPAWNS}"
                            """
                            sh "tar -cvz -f ${exp}_\$(date +%y%m%d_%H%M%S).tar.gz ./results/"
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