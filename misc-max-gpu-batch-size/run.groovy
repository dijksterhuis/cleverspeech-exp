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

        choice name: 'EXP_SCRIPT',
            choices: ['evasion_pgd', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'

        choice name: 'GRAPH_FILTER',
            choices: ['all', 'ctc', 'cw'],
            description: 'Filter experiments based on graph hyper parameter. Default: all.'

        choice name: 'EXP_SCRIPT',
            choices: ['evasion_pgd', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'

        choice name: 'SERVER_FILTER',
            choices: ['all', 'kalluto-0', 'kalluto-1', 'titan1-0', 'titan1-1', 'rhea-0'],
            description: 'Filter experiments based on gpu server parameter. Default: all.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'
    }
    environment {
        EXP_BASE_NAME = "conf-misc-max-gpu-batch-size"
        IMAGE = "dijksterhuis/cleverspeech:latest"
    }
    stages {
        stage("Modify jenkins build information") {
            steps {
                script {
                    buildName "#${BUILD_ID}: servers: ${params.SERVER_FILTER} script:${params.EXP_SCRIPT} data:${params.DATA} steps:${params.N_STEPS}"
                }
            }
        }
        stage("Create run commands"){

            steps {

                script {

                    def py_params = "--graph \${GRAPH}"
                    def container_params = "\${GRAPH}"

                    def py_cmd = """python3 ./experiments/${EXP_BASE_NAME}/${params.EXP_SCRIPT}.py \
                            --audio_indir ./samples/all/ \
                            ${params.ADDITIONAL_ARGS} \
                            ${py_params}"""

                    def container_name = "${EXP_BASE_NAME}-${BUILD_ID}-${container_params}"

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
        stage("Run all combos in parallel."){
            failFast false /* If one run fails, keep going! */
            matrix {
                axes {
                    axis {
                        name 'GRAPH'
                        values 'ctc', 'cw'
                    }
                    axis {
                        name 'SERVER'
                        values 'kalluto-0', 'kalluto-1', 'titan1-0', 'titan1-1', 'rhea-0'
                    }
                }
                agent {
                    label '\${SERVER}'
                }
                when {
                    anyOf {
                        allOf{
                            /* no filters applied so run everything */
                            expression { params.GRAPH_FILTER == 'all' }
                            expression { params.SERVER_FILTER == 'all' }
                        }
                        allOf {
                            /* exclusive filters applied, only run when all filters match */
                            expression { params.GRAPH_FILTER == env.GRAPH }
                            expression { params.SERVER_FILTER == env.SERVER }
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
