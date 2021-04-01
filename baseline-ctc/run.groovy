#!/usr/bin/env groovy

pipeline {
    /* Use jenkins build node to manage how many experiments to run at a time. */
    agent {
        label "build"
    }
    options {
        /* Don't need to do a version control checkout -- everything is in the docker image! */
        skipDefaultCheckout()
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    /*
    triggers {
        pollSCM('H H * * 1-5') }
        upstream(upstreamProjects: './build/latest', threshold: hudson.model.Result.SUCCESS) }
    }
    */
    environment {
        EXP_BASE_NAME = "baseline-ctc"
        IMAGE = "dijksterhuis/cleverspeech:latest"
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
            The next 4 parameters are used to determine how to run the attacks. These parameters are
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
        stage("Modify jenkins build information") {
            steps {
                script {
                    def name = "#${BUILD_ID}: type:${params.JOB_TYPE} script:${params.EXP_SCRIPT} data:${params.DATA} steps:${params.N_STEPS}"
                    buildName "${name}"
                }
            }
        }
        stage("Run combos in parallel."){
            failFast false /* If one run fails, keep going! */
            environment{
                /*
                Nasty way of not-really-but-sort-of simplifying the mess of our docker run command
                */
                DOCKER_NAME="${EXP_BASE_NAME}-\${LOSS}-${JOB_TYPE}"
                DOCKER_MOUNT="\$(pwd)/${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/"
                DOCKER_UID="LOCAL_UID=\$(id -u ${USER})"
                DOCKER_GID="LOCAL_GID=\$(id -g ${USER})"
                PYTHON_EXP="python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py \${LOSS}"
                PYTHON_ARG_1="--max_spawns ${params.MAX_SPAWNS}"
                PYTHON_ARG_2="--nsteps ${params.N_STEPS}"
                PYTHON_DATA_ARGS="--audio_indir ./${params.DATA}/all/ --targets_path ./${params.DATA}/cv-valid-test.csv"
                PYTHON_CMD = "${PYTHON_EXP} ${PYTHON_ARG_1} ${PYTHON_ARG_2} ${PYTHON_DATA_ARGS} ${params.ADDITIONAL_ARGS}"
            }
            matrix {
                /* Run each of these combinations over all axes on the gpu machines. */
                agent {
                    label "gpu"
                }
                axes {
                    axis {
                        name 'LOSS'
                        values 'ctc', 'ctc2'
                    }
                }
                stages {
                    stage("Pull docker image") {
                        steps {
                                sh "docker pull ${IMAGE}"
                        }
                    }

                    stage("Run experiment") {
                        when {
                            expression { params.JOB_TYPE == 'run' }
                        }
                        steps {
                            /* Run the attacks! */
                            sh  """
                                docker run \
                                    --gpus device=\${GPU_N} -t --rm --shm-size=10g --pid=host \
                                    --name ${DOCKER_NAME} \
                                    -v ${DOCKER_MOUNT} \
                                    -e ${DOCKER_UID} \
                                    -e ${DOCKER_GID} \
                                    ${IMAGE} \
                                    ${PYTHON_CMD}
                                """
                        }
                        post {
                            success {
                                archiveArtifacts artifacts: './${BUILD_ID}/', followSymlinks: false
                            }
                        }
                    }
                    stage("Run test") {
                        when {
                            expression { params.JOB_TYPE == 'test' }
                        }
                        steps {
                            sh  """
                                docker run \
                                    --gpus device=\${GPU_N} -t --rm --shm-size=10g --pid=host \
                                    --name ${DOCKER_NAME} \
                                    ${IMAGE} \
                                    ${PYTHON_CMD}
                                """
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
