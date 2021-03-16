#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    stages {
        stage("Test all the experiments!"){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'DIR'
                        values 'CTCBaselines', 'Confidence/CWMaxDiffBaselines', 'Confidence/AdaptiveKappa', 'Confidence/AlignmentEdgeCases', 'Confidence/InvertedCTC', 'Confidence/CumulativeLogProbs', 'Confidence/LogProbsGreedyDiff', 'Perceptual/Synthesis', 'Perceptual/RegularisedSynthesis', 'Perceptual/SpectralLossRegularisation'
                    }
                }
                stages {
                    stage("Test experiment") {
                        steps {
                            echo "Starting ${DIR} build job..."
                            build job: "${DIR}", wait: true
                        }
                    }
                }
            }
        }
    }
}