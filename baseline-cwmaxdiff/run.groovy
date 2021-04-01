#!/usr/bin/env groovy

pipeline {
    agent {
        label "build"
    }
    options {
        skipDefaultCheckout()
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    environment {
        EXP_BASE_NAME = "baseline-cwmaxdiff"
    }
    parameters {

            /*
            The first three parameters are priorities when deploying, as the "gpu" jenkins agents
            are shared resources and these parameters will determine whether jobs fail or not due to
            resource issues (basically, undergrads might be hogging GPU memory).
            */
            string name: 'MAX_SPAWNS',
                defaultValue: '3',
                description: 'Number of attacks to allow to spawn at once.'

            string name: 'N_STEPS',
                defaultValue: '10000',
                description: 'How many iterations to run the attack for.'

            string name: 'BATCH_SIZE',
                defaultValue: '10',
                description: 'How many examples in a batch.'

            /*
            The next 3 parameters are used to determine how to run the attacks. These parameters are
            generally sort-of hyper parameters (very sort of).
            */

            choice name: 'JOB_TYPE',
                choices: ['run', 'test'],
                description: 'Whether this is an experiment run or if we are just testing that everything works. default: run.'

            choice name: 'EXP_SCRIPT',
                choices: ['attacks', 'unbounded'],
                description: 'Which attack python script to run. default: attacks.py.'

            choice name: 'DATA',
                choices: ['samples', 'silence'],
                description: 'Which dataset to use. default: ./samples'

            text   name: 'ADDITIONAL_ARGS',
                defaultValue: '',
                description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'
    }

    stages {
        stage("Run experiments in parallel."){
            failFast false
            matrix {
                agent { label "gpu" }
                axes {
                    axis {
                        name 'ALIGNMENT'
                        values 'sparse', 'dense', 'ctcalign'
                    }
                    axis {
                        name 'DECODER'
                        values 'beam' /*, 'greedy'*/
                    }
                }
                stages {
                    stage("Run experiment") {
                        environment{
                            DOCKER_NAME="\${EXP_BASE_NAME}-\${ALIGNMENT}-\${DECODER}"
                            DOCKER_MOUNT="\$(pwd)/\${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/"
                            DOCKER_UID="LOCAL_UID=\$(id -u \${USER})"
                            DOCKER_GID="LOCAL_GID=\$(id -g \${USER})"
                            PYTHON_EXP="python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py \${ALIGNMENT}-\${DECODER}"
                            PYTHON_ARG_1="--max_spawns ${params.MAX_SPAWNS}"
                            PYTHON_ARG_2="--nsteps ${params.N_STEPS}"
                            PYTHON_DATA_ARGS="--audio_indir ./${params.DATA}/all/ --targets_path ./${params.DATA}/cv-valid-test.csv"
                        }
                        steps {
                            script {

                                def pythonArgs = "${PYTHON_EXP} ${PYTHON_ARG_1} ${PYTHON_ARG_2} ${params.ADDITIONAL_ARGS}"
                                buildDescription: "${params.JOB_TYPE}: ${pythonArgs}"
                                buildName: "#${BUILD_ID}-${params.JOB_TYPE}"

                                sh  """
                                    docker run \
                                        --gpus device=\${GPU_N} -t --rm --shm-size=10g --pid=host \
                                        --name ${DOCKER_NAME} \
                                        -v ${DOCKER_MOUNT} \
                                        -e ${DOCKER_UID} \
                                        -e ${DOCKER_GID} \
                                        dijksterhuis/cleverspeech:latest \
                                        ${PYTHON_EXP} \
                                        ${PYTHON_ARG_1} \
                                        ${PYTHON_ARG_2} \
                                        ${pythonArgs}
                                    """
                            }
                        }
                    }
                }
                post {
                    success {
                        archiveArtifacts artifacts: './${BUILD_ID}/', followSymlinks: false
                    }
                    always {
                        sh "docker container prune -f"
                        sh "docker image prune -f"
                    }
                }
            }
        }
    }
}
