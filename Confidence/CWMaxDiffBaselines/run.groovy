#!/usr/bin/env groovy

pipeline {
    agent { label "build" }
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
