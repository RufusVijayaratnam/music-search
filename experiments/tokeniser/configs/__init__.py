from .exp_base import experiments as base_experiments

def experiments():
  return {
      **base_experiments(),
  }
