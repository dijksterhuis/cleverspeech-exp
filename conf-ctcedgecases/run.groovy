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
    parameters {

        string name: 'N_STEPS',
            defaultValue: '10000',
            description: 'How many iterations to run the attack for.'

        string name: 'BATCH_SIZE',
            defaultValue: '10',
            description: 'How many examples in a batch.'

        choice name: 'JOB_TYPE',
            choices: ['run', 'test'],
            description: 'Whether this is an experiment run or if we are just testing that everything works. default: run.'

        choice name: 'EXP_SCRIPT',
            choices: ['attacks', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'

        choice name: 'DATA',
            choices: ['samples', 'silence'],
            description: 'Which dataset to use. default: ./samples'

        choice name: 'WRITER',
            choices: ['local_latest', 'local_all', 's3_latest', 's3_all'],
            description: 'How/where to write results data?. default: local.'

        choice name: 'ALIGNMENT_FILTER',
            choices: ['all', 'sparse', 'ctcalign', 'dense'],
            description: 'Filter experiments based on alignment hyper parameter. Default: batch.'

        /*choice name: 'LOSS_FILTER',
            choices: ['all', 'softmax', 'logits'],
            description: 'Filter experiments based on loss hyper parameter. Default: batch.'*/

        choice name: 'DECODER',
            choices: ['batch', 'greedy', 'batch_no_lm', 'greedy_no_lm'],
            description: 'decoder hyper parameter. Default: batch.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'
    }

    stages {
        stage("Modify jenkins build information") {
            steps {
                script {
                    buildName "#${BUILD_ID}: type:${params.JOB_TYPE} script:${params.EXP_SCRIPT} data:${params.DATA} steps:${params.N_STEPS}"
                }
            }
        }
        stage("Locked SCM checkout") {
            steps {
                lock("dummy") {
                    checkout scm
                }
            }
        }
        stage("Run all combos in parallel."){
            failFast false /* If one run fails, keep going! */
            matrix {
                agent {
                    label 'gpu'
                }
                axes {
                    /*axis {
                        name 'LOSS'
                        values 'softmax', 'logits'
                    }*/
                    axis {
                        name 'ALIGNMENT'
                        values 'sparse', 'ctcalign', 'dense'
                    }

                }
                when {
                    anyOf {
                        allOf{
                            /* no filters applied so run everything */
                            /*expression { params.LOSS_FILTER == 'all' }*/
                            expression { params.ALIGNMENT_FILTER == 'all' }
                        }
                        allOf {
                            /* exclusive filters applied, only run when all filters match */
                            /*expression { params.LOSS_FILTER == env.LOSS }*/
                            expression { params.ALIGNMENT_FILTER == env.ALIGNMENT }
                        }
                    }
                }
                environment {

                    EXP_BASE_NAME = "conf-ctcedgecases"
                    IMAGE = "dijksterhuis/cleverspeech:latest"

                    DOCKER_NAME="${EXP_BASE_NAME}-${BUILD_ID}-\${ALIGNMENT}"
                    DOCKER_MOUNT="\$(pwd)/${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/"
                    DOCKER_UID="LOCAL_UID=\$(id -u ${USER})"
                    DOCKER_GID="LOCAL_GID=\$(id -g ${USER})"

                    PY_BASE_CMD="python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py"
                    IN_DATA_ARG="--audio_indir ./${params.DATA}/all/"
                    TARGET_DATA_ARG="--targets_path ./${params.DATA}/cv-valid-test.csv"
                    OUTDIR_ARG="--outdir ./adv/${BUILD_ID}/${params.JOB_TYPE}"
                    STEPS_ARG="--nsteps ${params.N_STEPS}"
                    BATCH_ARG="--batch_size ${params.BATCH_SIZE}"
                    ALIGN_ARG="--align \${ALIGNMENT}"
                    /*LOSS_ARG="--loss ${params.LOSS}"*/
                    WRITER_ARG="--writer ${params.WRITER}"
                    DECODER_ARG="--decoder ${params.DECODER}"
                    PY_EXP_ARGS="${WRITER_ARG} ${BATCH_ARG} ${STEPS_ARG} ${ALIGN_ARG} ${DECODER_ARG}"

                    PYTHON_CMD = "${PY_BASE_CMD} ${PY_EXP_ARGS} ${IN_DATA_ARG} ${TARGET_DATA_ARG} ${OUTDIR_ARG} ${params.ADDITIONAL_ARGS}"

                }
                stages {
                    stage("Run experiment") {
                        when {
                            expression { params.JOB_TYPE == 'run' }
                        }
                        steps {
                            /* Run the attacks! */
                            sh  """
                                docker run \
                                    --pull=always \
                                    --gpus device=\${GPU_N} \
                                    -t \
                                    --rm \
                                    --shm-size=10g \
                                    --pid=host \
                                    --name ${DOCKER_NAME} \
                                    -v ${DOCKER_MOUNT} \
                                    -e ${DOCKER_UID} \
                                    -e ${DOCKER_GID} \
                                    -e AWS_ACCESS_KEY_ID=credentials('jenkins-aws-secret-key-id') \
                                    -e AWS_ACCESS_KEY_ID=credentials('jenkins-aws-secret-access-key') \
                                    ${IMAGE} \
                                    ${PYTHON_CMD}
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
                                    --pull=always \
                                    --gpus device=\${GPU_N} \
                                    -t \
                                    --rm \
                                    --shm-size=10g \
                                    --pid=host \
                                    --name ${DOCKER_NAME} \
                                    ${IMAGE} \
                                    ${PYTHON_CMD}
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
