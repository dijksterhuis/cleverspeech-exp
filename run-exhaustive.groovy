#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    options {
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    parameters {

        string name: 'BATCH_SIZE',
            defaultValue: '10',
            description: 'How many examples in a batch.'

        string name: 'N_STEPS',
            defaultValue: '25000',
            description: 'How many iterations to run the attack for.'
        /*
        choice name: 'EXP_SCRIPT',
            choices: ['attacks', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'

        choice name: 'DATA',
            choices: ['samples', 'silence'],
            description: 'Which dataset to use. default: ./samples'
        */

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'

    }
    stages {
        stage("Run all the experiments!"){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'DIR'
                        values 'baseline-ctc',
                            'baseline-cwmaxdiff',
                            'baseline-biggiomaxmin',
                            'conf-adaptivekappa',
                            'conf-ctcedgecases',
                            'conf-invertedctc',
                            'conf-logprobsgreedydiff',
                            'conf-sumlogprobs',
                            'conf-cumulativelogprobs',
                            'conf-biggiomaxofmaxmin',
                            'conf-targetonly',
                            'misc-batch-vs-indy' /*,
                             'percep-synthesis',
                             'percep-synthesisregularised',
                             'percep-spectralloss'
                             */
                    }
                    axis {
                        name 'MATRIX_DATA'
                        values 'samples', 'silence'
                    }
                    axis {
                        name 'MATRIX_EXP_SCRIPT'
                        values 'attacks', 'unbounded'
                    }
                }
                stages {
                    stage("Run experiment") {
                        steps {
                            echo "Starting ${DIR} build job..."
                            build job: "${DIR}",
                                wait: true,
                                parameters: [
                                    stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                                    stringParam(name: 'EXP_SCRIPT', value: "${MATRIX_EXP_SCRIPT}"),
                                    stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                                    stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                                    stringParam(name: 'DATA', value: "${MATRIX_DATA}"),
                                    stringParam(name: 'JOB_TYPE', value: "run"),
                                ]
                        }
                    }
                }
            }
        }
    }
}