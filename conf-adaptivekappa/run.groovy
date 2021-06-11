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
            choices: ['all', 'sparse', 'mid', 'dense', 'ctc'],
            description: 'Filter experiments based on alignment hyper parameter. Default: batch.'

        /*choice name: 'LOSS_FILTER',
            choices: ['all', 'softmax', 'logits'],
            description: 'Filter experiments based on loss hyper parameter. Default: batch.'*/

        choice name: 'DECODER',
            choices: ['batch', 'greedy', 'batch_no_lm', 'greedy_no_lm', 'tf_greedy', 'tf_beam'],
            description: 'decoder hyper parameter. Default: batch.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'
    }
    environment {
        EXP_BASE_NAME = "conf-adaptivekappa"
        IMAGE = "dijksterhuis/cleverspeech:latest"
    }
    stages {
        stage("Modify jenkins build information") {
            steps {
                script {
                    buildName "#${BUILD_ID}: type:${params.JOB_TYPE} script:${params.EXP_SCRIPT} data:${params.DATA} steps:${params.N_STEPS}"
                }
            }
        }
        stage("Create run commands"){

            steps {

                script {

                    def py_params = "--align \${ALIGNMENT}" /* --loss \${LOSS}"*/
                    def container_params = "\${ALIGNMENT}" /*"\${LOSS}-*/

                    def py_cmd = """python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py \
                            --audio_indir ./${params.DATA}/all/ \
                            --targets_path ./${params.DATA}/cv-valid-test.csv \
                            --outdir ./adv/${BUILD_ID}/${params.JOB_TYPE} \
                            --nsteps ${params.N_STEPS} \
                            --batch_size ${params.BATCH_SIZE} \
                            --writer ${params.WRITER} \
                            --decoder ${params.DECODER} \
                            ${params.ADDITIONAL_ARGS} \
                            ${py_params}"""

                    def container_name = "${EXP_BASE_NAME}-${BUILD_ID}-${params.JOB_TYPE}-${container_params}"

                    if (params.JOB_TYPE == "run") {
                        if (params.WRITER == "s3_latest" || params.WRITER == "s3_all") {

                            def cmd = """
                                    docker run \
                                        --pull=always \
                                        --gpus device=\${GPU_N} \
                                        -t \
                                        --rm \
                                        --shm-size=10g \
                                        --pid=host \
                                        --name ${container_name} \
                                        -v \$(pwd)/${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/ \
                                        -e LOCAL_UID=\$(id -u ${USER}) \
                                        -e LOCAL_GID=\$(id -g ${USER}) \
                                        -e AWS_ACCESS_KEY_ID=\"\${AWS_ID}\" \
                                        -e AWS_SECRET_ACCESS_KEY=\"\${AWS_SECRET}\" \
                                        ${IMAGE} ${py_cmd}
                            """
                            CMD = "${cmd}"
                        }
                        else if (params.WRITER == "local_latest" || params.WRITER == "local_all") {

                            def cmd = """
                                    docker run \
                                        --pull=always \
                                        --gpus device=\${GPU_N} \
                                        -t \
                                        --rm \
                                        --shm-size=10g \
                                        --pid=host \
                                        --name ${container_name} \
                                        -v \$(pwd)/${BUILD_ID}:/home/cleverspeech/cleverSpeech/adv/ \
                                        -e LOCAL_UID=\$(id -u ${USER}) \
                                        -e LOCAL_GID=\$(id -g ${USER}) \
                                        ${IMAGE} ${py_cmd}
                            """
                            CMD = "${cmd}"
                        }
                    }
                    else if (params.JOB_TYPE == "test") {
                        def cmd = """
                                docker run \
                                    --pull=always \
                                    --gpus device=\${GPU_N} \
                                    -t \
                                    --rm \
                                    --shm-size=10g \
                                    --pid=host \
                                    --name ${container_name} \
                                    ${IMAGE} ${py_cmd}

                        """
                        CMD = "${cmd}"
                    }
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
                        values 'sparse', 'mid', 'dense', 'ctc'
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
                stages {
                    stage("exe") {
                        environment {
                            AWS_ID = credentials('jenkins-aws-secret-key-id')
                            AWS_SECRET = credentials('jenkins-aws-secret-access-key')
                        }
                        steps {
                            sh  "${CMD}"
                        }
                    }
                    stage("res") {
                        when {
                            expression { params.JOB_TYPE == 'run' }
                        }
                        steps {
                            archiveArtifacts "${BUILD_ID}/**"
                        }
                    }
                }
                post {
                    always {
                        lock("docker cleanup") {
                            sh "docker container prune -f"
                            sh "docker image prune -f"
                        }
                    }
                }
            }
        }
    }
}
