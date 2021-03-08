#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Baselines/"
        CLEVERSPEECH_HOME = "/home/cleverspeech/cleverSpeech"
    }
    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'experiment'
                        values 'ctc', 'ctc_v2', 'ctcalign_maxdiff_greedy', 'ctcalign_maxdiff_beam'
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
                        steps {
                            script {
                                echo "+=+=+=+=+=====> Running experiment: ${experiment}"
                                def exp = "${experiment}"
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
