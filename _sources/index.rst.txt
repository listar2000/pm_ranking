.. PM-RANK documentation master file, created by
   sphinx-quickstart on Thu Jul 24 08:26:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PM-RANK Documentation
=====================

Welcome to the documentation for :code:`PM_RANK`: an analysis toolkit for prediction markets.

In the beginning, we develop the :code:`pm_rank` package to support our :code:`Prophet Arena` `platform <https://www.prophetarena.co/>`_. 

.. image:: _static/prophet_arena_web.png
   :width: 100%
   :align: center

But since
:code:`pm_rank` provides a **unified, hierarchical interface** for defining prediction market and its events, we expect it to be useful for broader audience, 
especially for those who want to integrate handy scoring/ranking algorithms into their own projects.

Below we will provide a quick overview of core concepts (e.g. models, dataclass interfaces, etc.) in :code:`pm_rank`. Please refer to:

- :doc:`autoapi/src/pm_rank/index` for **detailed API documentation**.
- `Colab Demo <https://colab.research.google.com/drive/1gAXhNQySdCP1L9HjVQA8vyLz5393pjdI?usp=sharing>`_ for a **quick demo** using the package to load prediction market data and obtain some interesting insights.
- :doc:`blogpost/ranking_llm_250727` for our detailed blogpost **walking through the reasoning behind our ranking module design**.

.. toctree::
   :maxdepth: 2
   :caption: Links:
   :hidden:

   colab_demo
   autoapi/src/pm_rank/index

.. toctree::
   :maxdepth: 2
   :caption: Blogposts:
   :hidden:

   blogpost/ranking_llm_250727

Quick Installation Guide
------------------------

:code:`pm_rank` is a python package that can be installed via :code:`pip` (requires python 3.8 or higher)

.. code-block:: bash

    pip install -U pm_rank

The default version uses minimal dependencies (i.e. no :code:`pytorch`), so some ranking models (e.g. :code:`IRT`) are not available.

To install the full version, you can install the :code:`full` dependency::

    pip install -U pm_rank[full]

.. note::
   **For potential developers:**  
   If you want to contribute to the documentation, you can install the :code:`docs` dependency:

   .. code-block:: bash

      pip install -U pm_rank[docs]

   Then you can build the documentation by running:

   .. code-block:: bash

      cd docs && make html

Core Concepts
-------------

We first use a flowchart to illustrate the pipeline of using :code:`pm_rank`:

.. image:: _static/package_overview.png
   :width: 100%
   :align: center

In a nutshell, at the beginning of this pipeline, user can choose to bring their own dataset (e.g. from human prediction market platforms) and implement
their own subclass that inherits from :code:`ChallengeLoader` (see :doc:`autoapi/src/pm_rank/data/loaders/index`). 

Once this is done (i.e. the loader implements the standard :code:`.load_challenge()` method), the downstream steps are **data-source independent** and we
introduce the core concepts here:

.. note::
   Please refer to :doc:`autoapi/src/pm_rank/data/base/index` for the actual data model implementation. We give a high-level and non-comprehensive overview in a **bottom-up** manner.

1. **ForecastEvent**: this is the most atomic unit of prediction market data. It represents a single prediction made by a forecaster for a single forecast problem.

   **Key Fields in ForecastEvent:**

   * ``problem_id``: an unique identifier for the problem
   * ``username``: an unique identifier for the forecaster
   * ``timestamp``: the timestamp of the prediction. Note that this is not optional as we might want to **stream** the predictions in time. However, if the original data does not contain this information, we will use the current time as a placeholder.
   * ``probs``: the probability distribution over the options -- given by the forecaster.
   * ``unnormalized_probs``: the unnormalized probability distribution over the options -- given by the forecaster.

2. **ForecastProblem**: this is a collection of ``ForecastEvent``\s for a single forecast problem. It validates keeps track of metadata for the problem like the options and the correct option. It is also a handy way to organize the dataset as we treat ``ForecastProblem`` as the basic unit of **streaming prediction market data**.

   In particular, if a ``ForecastProblem`` has the ``odds`` field, we would answer questions like "how much money can an individual forecaster make" and use these results to rank the forecasters. See :doc:`autoapi/src/pm_rank/model/average_return/index` for more details.

   **Key Fields in ForecastProblem:**

   * ``title``: the title of the problem
   * ``problem_id``: the id of the problem
   * ``options``: the options for the problem
   * ``correct_option_idx``: the index of the correct option
   * ``forecasts``: the forecasts for the problem
   * ``num_forecasters``: the number of forecasters
   * ``url``: the URL of the problem
   * ``odds`` (optional): the market odds for each option

3. **ForecastChallenge**: this is a collection of ``ForecastProblem``\s. It implements two **core functionalities for all scoring/ranking methods** to use:

   * ``get_problems -> List[ForecastProblem]``: return all the problems in the challenge. Suitable for the *full-analysis* setting.
   * ``stream_problems -> Iterator[List[ForecastProblem]]``: return the problems in the challenge in a streaming setting. This setting **simulates the real-world scenario** where the predictions enter gradually. The scoring/ranking methods can also leverage this function to efficiently calculate the metrics at different time points (batches).








