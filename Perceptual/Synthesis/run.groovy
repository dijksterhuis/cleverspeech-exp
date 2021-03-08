#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE = "dijksterhuis/cleverspeech:latest"
        EXP_DIR = "./experiments/Perceptual/Synthesis"
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
        stage("Run experiments."){
            steps {
                script {
                    def experiments = ['freq_harmonic', 'full_harmonic', 'inharmonic', 'detnoise_freq_harmonic', 'detnoise_full_harmonic', 'detnoise_inharmonic', 'stft']
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