#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Baselines/"
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
        stage("Run Sparse Align experiments."){
            steps {
                script {
                    def experiments = ['ctc', 'ctc_v2', 'ctcalign_maxdiff_greedy', 'ctcalign_maxdiff_beam']
                    for (int i = 0; i < experiments.size(); ++i) {
                        echo "Running ${experiments[i]}"
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
                                python3 ${EXP_DIR}/attacks.py ${experiments[i]} --max_spawns 5
                        """
                    }

                }
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