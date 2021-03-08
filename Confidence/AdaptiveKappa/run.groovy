#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Confidence/AdaptiveKappa/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'alignment_type'
                        values 'dense', 'sparse', 'ctcalign'
                    }
                    axis {
                        name 'loss_type'
                        values '', '-rctc', '-ctc'
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
                            echo "+=+=+=+=+=====> Running experiment: ${alignment_type}${loss_type}"
                            def exp = "${alignment_type}${loss_type}"
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
        }
    }
}