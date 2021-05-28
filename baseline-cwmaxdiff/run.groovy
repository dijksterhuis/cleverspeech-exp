#!/usr/bin/env groovy

pipeline {
    /* Use jenkins build node to manage how many experiments to run at a time. */
    agent {
        label "build"
    }
    options {
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
        EXP_BASE_NAME = "baseline-cwmaxdiff"
        IMAGE = "dijksterhuis/cleverspeech:latest"
    }
    parameters {

            /*
            The first three parameters are priorities when deploying, as the "gpu" jenkins agents
            are shared resources and these parameters will determine whether jobs fail or not due to
            resource issues (basically, undergrads might be hogging GPU memory).
            */


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

            choice name: 'ALIGNMENT_FILTER',
                choices: ['all', 'dense', 'ctcalign', 'sparse'],
                description: 'Filter experiments based on alignment hyper parameter. Note that this only works when combined with other filters.'

            choice name: 'LOSS_FILTER',
                choices: ['all', 'softmax', 'logits'],
                description: 'Filter experiments based on loss hyper parameter. Note that this only works when combined with other filters.'

            choice name: 'DECODER_FILTER',
                choices: ['all', 'batch', 'greedy'],
                description: 'Filter experiments based on decoder hyper parameter. Note that this only works when combined with other filters.'

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
        stage("Locked SCM checkout") {
            steps {
                lock("dummy") {
                    sleep 5
                    checkout scm
                }
            }
        }
        stage("Run combos in parallel."){
            failFast false /* If one run fails, keep going! */
            environment{
                /*
                Nasty way of not-really-but-sort-of simplifying the mess of our docker run command
                */
                DOCKER_NAME="${EXP_BASE_NAME}-${EXP_SCRIPT}-${DATA}-\${ALIGNMENT}-\${DECODER}-\${LOSS}-${JOB_TYPE}"
                DOCKER_MOUNT="\$(pwd)/${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/"
                DOCKER_UID="LOCAL_UID=\$(id -u ${USER})"
                DOCKER_GID="LOCAL_GID=\$(id -g ${USER})"
                AWS_ACCESS_KEY_ID = credentials('jenkins-aws-secret-key-id')
                AWS_SECRET_ACCESS_KEY = credentials('jenkins-aws-secret-access-key')

                PY_BASE_CMD="python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py"
                PY_DATA_ARGS="--audio_indir ./${params.DATA}/all/ --targets_path ./${params.DATA}/cv-valid-test.csv"


                STEPS_ARG="--nsteps ${params.N_STEPS}"
                BATCH_ARG="--batch_size ${params.BATCH_SIZE}"
                ALIGN_ARG="--align \${ALIGNMENT}"
                LOSS_ARG="--loss \${LOSS}"
                DECODER_ARG="--decoder \${DECODER}"
                PY_EXP_ARGS="${BATCH_ARG} ${BATCH_ARG} ${STEPS_ARG} ${ALIGN_ARG} ${DECODER_ARG} ${LOSS_ARG}"

                PYTHON_CMD = "${PY_BASE_CMD} ${PY_EXP_ARGS} ${PY_DATA_ARGS} ${params.ADDITIONAL_ARGS}"
            }
            matrix {
                /* Run each of these combinations over all axes on the gpu machines. */
                agent {
                    label "gpu"
                }
                when {
                    anyOf {
                        allOf{
                            /* no filters applied so run everything */
                            expression { params.ALIGNMENT_FILTER == 'all' }
                            expression { params.LOSS_FILTER == 'all' }
                            expression { params.DECODER_FILTER == 'all' }
                        }
                        allOf {
                            /* exclusive filters applied, only run when all filters match */
                            expression { params.ALIGNMENT_FILTER == env.ALIGNMENT }
                            expression { params.LOSS_FILTER == env.LOSS }
                            expression { params.DECODER_FILTER == env.DECODER }
                        }
                    }
                }
                axes {
                    axis {
                        name 'ALIGNMENT'
                        values 'sparse', 'dense', 'ctcalign'
                    }
                    axis {
                        name 'DECODER'
                        values 'batch', 'greedy'
                    }
                    axis {
                        name 'LOSS'
                        values 'softmax', 'logits'
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
                                    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
                                    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
                                    ${IMAGE} \
                                    ${PYTHON_CMD} --writer s3
                                """
                                archiveArtifacts "${BUILD_ID}/**"
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
                                    ${PYTHON_CMD} --writer local
                                """
                        }
                    }
                }
                post {
                    always {
                        sh "docker image prune -f"
                    }
                }
            }
        }
    }
}
