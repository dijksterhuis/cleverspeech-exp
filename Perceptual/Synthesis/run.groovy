#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
    options { skipDefaultCheckout() }
    parameters {
            string(name: 'MAX_SPAWNS', defaultValue: '5', description: 'Number of attacks to spawn at once.')
        }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Perceptual/Synthesis/"
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
                        values 'additive', 'detnoise'
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
                            echo "+=+=+=+=+=====> Running experiment: ${detnoise}-${synth}"
                            def exp = "${detnoise}-${synth}"
                            sh """
                                docker run \
                                    --gpus device=${GPU_N} \
                                    -t \
                                    --rm \
                                    --shm-size=10g \
                                    --name ${exp} \
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
                        python3 ${EXP_DIR}/attacks.py stft --max_spawns "${params.MAX_SPAWNS}"
                """
                sh "tar -cvz -f stft_\$(date +%y%m%d_%H%M%S).tar.gz ./results/"
                sh "docker container prune -f"
                sh "docker image prune -f"
            }
        }
    }
}
