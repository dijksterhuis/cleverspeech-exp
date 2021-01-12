FROM dijksterhuis/cleverspeech:build

# move to the latest version of the repo's relevant branch

ARG BRANCH=master
ENV BRANCH=${BRANCH}

RUN git checkout "${BRANCH}" && git pull origin "${BRANCH}"

