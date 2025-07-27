{{ name | replace('src.pm_rank.', 'pm_rank/') | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   :inherited-members:

{% block attributes %}
{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
   :toctree:
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
{% for item in functions %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
{% for item in classes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
   :toctree:
{% for item in exceptions %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
{% for item in modules %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %} 