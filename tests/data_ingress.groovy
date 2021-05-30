#!/usr/bin/env groovy

pipeline {
    agent { label 'cpu' }
    options {
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    triggers {
        upstream(upstreamProjects: '../0-build/latest', threshold: hudson.model.Result.SUCCESS)
    }
    parameters {

        string name: 'MAX_EXAMPLES',
            defaultValue: '1000',
            description: 'How many examples.'

        string name: 'BATCH_SIZE',
            defaultValue: '100',
            description: 'How many examples in a batch.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10.'

    }
    stages {
        stage("Tests"){
            failFast false
            matrix {
                agent {
                    label 'gpu'
                }
                environment {
                    IMAGE = "dijksterhuis/cleverspeech:latest"
                }
                axes {
                    axis {
                        name 'ETL'
                        values 'standard', 'sparse', 'dense'
                    }
                    axis {
                        name 'DATA'
                        values 'samples', 'silence'
                    }
                }
                stages {
                    stage("t") {
                        steps {
                            sh  """
                                docker run \
                                    --pull=always \
                                    --gpus device=\${GPU_N} \
                                    -t \
                                    --rm \
                                    --name etl-ingress-test-\${ETL}-\${DATA} \
                                    ${IMAGE} \
                                        python3 \
                                        ./experiments/tests/pytests/test_data.py \
                                        --audio_indir ./${DATA}/all/ \
                                        --etl ${ETL} \
                                        --max_examples params.MAX_EXAMPLES \
                                        --batch_size params.BATCH_SIZE
                                """
                        }
                    }
                }
            }
        }
    }
}