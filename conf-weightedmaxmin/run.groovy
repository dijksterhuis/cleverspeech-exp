#!/usr/bin/env groovy

pipeline {
    /* Use jenkins build node to manage how many experiments to run at a time. */
    agent {
        label "gpu"
    }
    options {
        skipDefaultCheckout()
        timestamps()
        disableResume()
    }
    /*
    triggers {
        pollSCM('H H * * 1-5') }
        upstream(upstreamProjects: './build/latest', threshold: hudson.model.Result.SUCCESS) }
    }
    */
    environment {

        EXP_BASE_NAME = "conf-weightedmaxmin"
        IMAGE = "dijksterhuis/cleverspeech:latest"

        DOCKER_NAME="${EXP_BASE_NAME}-${BUILD_ID}"
        DOCKER_MOUNT="\$(pwd)/${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/"
        DOCKER_UID="LOCAL_UID=\$(id -u ${USER})"
        DOCKER_GID="LOCAL_GID=\$(id -g ${USER})"
        AWS_ACCESS_KEY_ID = credentials('jenkins-aws-secret-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('jenkins-aws-secret-access-key')

        PY_BASE_CMD="python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py"
        IN_DATA_ARG="--audio_indir ./${params.DATA}/all/"
        TARGET_DATA_ARG="--targets_path ./${params.DATA}/cv-valid-test.csv"
        OUTDIR_ARG="--outdir ./adv/${BUILD_ID}/${params.JOB_TYPE}"
        STEPS_ARG="--nsteps ${params.N_STEPS}"
        BATCH_ARG="--batch_size ${params.BATCH_SIZE}"
        ALIGN_ARG="--align ${params.ALIGNMENT}"
        DECODER_ARG="--decoder ${params.DECODER}"
        WRITER_ARG="--writer ${params.WRITER}"
        PY_EXP_ARGS="${WRITER_ARG} ${BATCH_ARG} ${BATCH_ARG} ${STEPS_ARG} ${ALIGN_ARG} ${DECODER_ARG}"

        PYTHON_CMD = "${PY_BASE_CMD} ${PY_EXP_ARGS} ${IN_DATA_ARG} ${TARGET_DATA_ARG} ${OUTDIR_ARG} ${params.ADDITIONAL_ARGS}"

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

        choice name: 'WRITER',
            choices: ['local', 's3'],
            description: 'How/where to write results data?. default: local.'

        choice name: 'ALIGNMENT',
            choices: ['sparse', 'dense', 'ctcalign'],
            description: 'Filter experiments based on alignment hyper parameter. Default: sparse.'

        choice name: 'LOSS',
            choices: ['softmax', 'logits'],
            description: 'Filter experiments based on loss hyper parameter. Default: softmax.'

        choice name: 'DECODER',
            choices: ['batch', 'greedy', 'batch_no_lm', 'greedy_no_lm'],
            description: 'Filter experiments based on decoder hyper parameter. Default: batch.'

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
                    sleep 5
                    checkout scm
                }
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
                        -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
                        -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
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