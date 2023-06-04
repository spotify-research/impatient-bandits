# impatient-bandits

Companion code for the paper:

> Thomas M. McDonald, Lucas Maystre, Mounia Lalmas, Daniel Russo, Kamil Ciosek.
> [_Impatient Bandits: Optimizing Recommendations for the Long-Term Without
> Delay_](#). Proceedings of KDD 2023.

This repository contains

- a reference implementation of the algorithms presented in the paper, and
- Jupyter notebooks providing experiments on synthetic data similar to the ones
  presentd in the paper.

The paper addresses the problem of optimizing a sequence of decisions for
long-term rewards. Assuming that intermediate outcomes correlated with the
final reward are revealed progressively over time, we provide a method that is
able to take advantage of these intermediate observations effectively.
<!-- For an accessible overview of the main idea, you can read our [blog
post](#).-->

## Getting Started

To get started, follow these steps:

- Clone the repo locally with: `git clone
  https://github.com/spotify-research/impatient-bandits.git`
- Move to the repository: `cd impatient-bandits`
- Install the dependencies: `pip install -r requirements.txt`
- Install the package: `pip install -e lib/`
- Move to the notebook folder: `cd notebooks`
- Start a notebook server: `jupyter notebok`

Our codebase was tested with Python 3.11.3. The following libraries are required
(and installed automatically via the first `pip` command above):

- notebook (tested with version 6.5.4)
- matplotlib (tested with version 3.7.1)
- numpy (tested with version 1.25.0)
- scipy (tested with version 1.10.1)

## Support

Create a [new issue](https://github.com/spotify-research/impatient-bandits/issues/new)

## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](https://github.com/spotify/code-of-conduct/blob/main/code-of-conduct.md)
in all interactions with the community.

## Authors

- [Tom McDonald](mailto:tommcdonald955@gmail.com )
- [Lucas Maystre](mailto:lucasm@spotify.com)
- [Kamil Ciosek](mailto:kamilc@spotify.com)

A full list of [contributors](https://github.com/spotify-research/impatient-bandits/graphs/contributors?type=a) can be found on GHC

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for updates.

## License

Copyright 2023 Spotify, Inc.

Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program (https://hackerone.com/spotify) rather than GitHub.
