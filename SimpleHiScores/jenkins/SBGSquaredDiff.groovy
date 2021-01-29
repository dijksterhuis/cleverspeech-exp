#!/usr/bin/env groovy

pipeline {
    agent { label "gpu" }
    environment {
        IMAGE_NAME = "dijksterhuis/cleverspeech"
        TAG = "latest"
        EXP_ARG = "squared_diff_loss"
        SEARCH_CMD = "./experiments/StrongAlignments/targeting/simple_branched_grid_search.py"
        CLEVERSPEECH_HOME=/home/cleverspeech/cleverSpeech
    }

    stages {

        stage('Prep work.') {
            steps {
                script {
                    sh "docker container prune -f"
                    withDockerRegistry([ credentialsId: "dhub-mr", url: "" ]) {
                        sh "docker pull ${IMAGE_NAME}:${TAG}"
                    }
                }
            }
        }

        stage("Generate Target Logits data with a simple branched grid search."){
            steps {
                script {

                    sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name ${EXP_ARG} \
                        -v \$(pwd)/target-logits/:${CLEVERSPEECH_HOME}/adv \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE_NAME}:${TAG} \
                        python3 \
                        ${SEARCH_CMD}
                    """
                }
            }
        }

        stage("Run the Custom Squared Difference Loss attack."){
            steps {
                script {

                    sh """
                    docker run \
                        --gpus device=${GPU_N} \
                        -t \
                        --rm \
                        --name ${EXP_ARG} \
                        -v \$(pwd)/target-logits/:${CLEVERSPEECH_HOME}/target-logits/ \
                        -v \$(pwd)/results/:${CLEVERSPEECH_HOME}/adv/ \
                        -e LOCAL_UID=\$(id -u ${USER}) \
                        -e LOCAL_GID=\$(id -g ${USER}) \
                        ${IMAGE_NAME}:${TAG} \
                        python3 \
                        ./experiments/StrongAlignments/attacks.py \
                        ${EXP_ARG} \
                        --max_spawns 5
                    """
                }
            }
        }
    }
    post  {
        always {
            sh "docker image rm ${IMAGE_NAME}:${TAG}"
            sh "docker image prune -f"
            sh "docker container prune -f"
        }
    }
}