#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Perceptual/RegularisedSynthesis/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Prep work.") {
            steps {
                script {
                    withDockerRegistry([ credentialsId: "dhub-mr", url: "" ]) {
                        sh "docker container prune -f"
                        sh "docker pull ${IMAGE}"
                    }
                }
            }
        }
        stage("Run experiments in parallel."){
            failFast false
            matrix {
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
                    stage("Prep work.") {
                        steps {
                            script {
                                withDockerRegistry([ credentialsId: "dhub-mr", url: "" ]) {
                                    sh "docker container prune -f"
                                    sh "docker pull ${IMAGE}"
                                }
                            }
                        }
                    }
                    stage("Run experiment") {
                        script {
                            echo "+=+=+=+=+=====> Running experiment: ${synth}${detnoise}"
                            def exp = "${synth}${detnoise}"
                            sh """
                                docker run \
                                    --gpus device=${GPU_N} \
                                    -t \
                                    --rm \
                                    --name ${EXP_ARG} \
                                    -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                                    -e LOCAL_UID=\$(id -u ${USER}) \
                                    -e LOCAL_GID=\$(id -g ${USER}) \
                                    ${IMAGE} \
                                    python3 ${EXP_DIR}/attacks.py ${exp} --max_spawns 5
                            """
                        }
                    }
                }
                post {
                    always {
                        sh "docker image prune -f"
                        sh "docker container prune -f"
                        sh "docker image rm ${IMAGE}"
                    }
                }
            }
        stage("STFT Prep work.") {
            steps {
                script {
                    withDockerRegistry([ credentialsId: "dhub-mr", url: "" ]) {
                        sh "docker container prune -f"
                        sh "docker pull ${IMAGE}"
                    }
                }
            }
                    }
        stage("Run STFT experiment") {
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
                        ${IMAGE} \
                        python3 ${EXP_DIR}/attacks.py stft --max_spawns 5
                """
            }
        }
    }
    post  {
        always {
            sh "docker image prune -f"
            sh "docker container prune -f"
            sh "docker image rm ${IMAGE}"
        }
    }
}
