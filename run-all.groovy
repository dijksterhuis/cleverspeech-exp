#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    options {
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    /*
    triggers {
        not applicable
    }
    */
    parameters {
        string name: 'BATCH_SIZE',
            defaultValue: '10',
            description: 'How many examples in a batch.'

        string name: 'N_STEPS',
            defaultValue: '25000',
            description: 'How many iterations to run the attack for.'

        choice name: 'EXP_SCRIPT',
            choices: ['evasion_pgd', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'

            choice name: 'DATA',
                choices: ['samples', 'silence', 'reference-signals/sines/sine', 'reference-signals/sines/am', 'reference-signals/sines/fm', 'reference-signals/sines/additive', 'reference-signals/noise/uniform'],
                description: 'Which dataset to use. default: ./samples'

            choice name: 'WRITER',
            choices: ['local_latest', 'local_all', 's3_latest', 's3_all'],
                description: 'How/where to write results data?. default: local.'

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
                }
                stages {
                    stage("Run experiment") {
                        steps {
                            echo "Starting ${DIR} build job..."
                            build job: "${DIR}",
                                wait: true,
                                parameters: [
                                    stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                                    stringParam(name: 'EXP_SCRIPT', value: "${params.EXP_SCRIPT}"),
                                    stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                                    stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                                    stringParam(name: 'WRITER', value: "${params.WRITER}"),
                                    stringParam(name: 'DATA', value: "${params.DATA}"),
                                    stringParam(name: 'JOB_TYPE', value: "run"),
                                ]
                        }
                    }
                }
            }
        }
    }
}