_SYSTEM_CONFIG_SWORDSHIELD = """
bodies {
  name: "pelvis"
  colliders {
    position {
      z: 0.07
    }
    sphere {
      radius: 0.09
    }
  }
  colliders {
    position {
      z: 0.205
    }
    sphere {
      radius: 0.07
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 10.0
}
bodies {
  name: "torso"
  colliders {
    position {
      z: 0.12
    }
    sphere {
      radius: 0.11
    }
  }
  colliders {
    position {
      x: -0.01142375
      y: -0.08697725
      z: 0.23320685
    }
    rotation {
      x: 83.8885
      y: -7.4404535
      z: -6.6881237
    }
    capsule {
      radius: 0.045
      length: 0.17357418
    }
  }
  colliders {
    position {
      x: -0.01142375
      y: 0.08697725
      z: 0.23320685
    }
    rotation {
      x: -83.8885
      y: -7.4404535
      z: 6.6881237
    }
    capsule {
      radius: 0.045
      length: 0.17357418
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 12.002052
}
bodies {
  name: "head"
  colliders {
    position {
      z: 0.175
    }
    sphere {
      radius: 0.095
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.8822644
}
bodies {
  name: "right_upper_arm"
  colliders {
    position {
      z: -0.14
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.045
      length: 0.27
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.4993314
}
bodies {
  name: "right_lower_arm"
  colliders {
    position {
      z: -0.12
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.04
      length: 0.215
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.9996799
}
bodies {
  name: "right_hand"
  colliders {
    position {
    }
    sphere {
      radius: 0.04
    }
  }
  colliders {
    position {
      x: -0.155
    }
    rotation {
      x: -0.0
      y: 90.0
    }
    capsule {
      radius: 0.023
      length: 0.276
    }
  }
  colliders {
    position {
      x: 0.4
    }
    rotation {
      y: -0.0
    }
    box {
      halfsize {
        x: 0.34
        y: 0.01
        z: 0.035
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2
}
bodies {
  name: "left_upper_arm"
  colliders {
    position {
      z: -0.14
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.045
      length: 0.27
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.4993314
}
bodies {
  name: "left_lower_arm"
  colliders {
    position {
      z: -0.12
    }
    rotation {
      y: -0.0
    }
    capsule {
      radius: 0.04
      length: 0.215
    }
  }
  colliders {
    position {
      y: 0.085
      z: -0.12
    }
    rotation {
      x: -90.0
    }
    box {
      halfsize {
        x: 0.266
        y: 0.266
        z: 0.015
      }
    }
  }
  colliders {
    position {
      z: -0.258947
    }
    sphere {
      radius: 0.04
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 3.12
}
bodies {
  name: "right_thigh"
  colliders {
    position {
      z: -0.21
    }
    rotation {
    }
    capsule {
      radius: 0.055
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5022917
}
bodies {
  name: "right_shin"
  colliders {
    position {
      z: -0.2
    }
    rotation {
    }
    capsule {
      radius: 0.05
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.9997497
}
bodies {
  name: "right_foot"
  colliders {
    position {
      x: 0.045
      z: -0.0225
    }
    rotation {
    }
    box {
      halfsize {
        x: 0.0885
        y: 0.045
        z: 0.0275
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.99968714
}
bodies {
  name: "left_thigh"
  colliders {
    position {
      z: -0.21
    }
    rotation {
    }
    capsule {
      radius: 0.055
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5022917
}
bodies {
  name: "left_shin"
  colliders {
    position {
      z: -0.2
    }
    rotation {
    }
    capsule {
      radius: 0.05
      length: 0.41
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.9997497
}
bodies {
  name: "left_foot"
  colliders {
    position {
      x: 0.045
      z: -0.0225
    }
    rotation {
    }
    box {
      halfsize {
        x: 0.0885
        y: 0.045
        z: 0.0275
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.99968714
}
bodies {
    name: "floor"
    colliders {
      plane {
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
}
joints {
  name: "spine1"
  parent: "pelvis"
  child: "torso"
  parent_offset {
    z: 0.236151
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 100.0
}
joints {
  name: "neck"
  parent: "torso"
  child: "head"
  parent_offset {
    z: 0.223894
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 10.0
}
joints {
  name: "right_shoulder"
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    x: -0.02405
    y: -0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 40.0
}
joints {
  name: "right_elbow"
  parent: "right_upper_arm"
  child: "right_lower_arm"
  parent_offset {
    z: -0.274788
  }
  child_offset {
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -160.0
    max: 0.0
  }
  angular_damping: 30.0
}
joints {
  name: "right_wrist"
  parent: "right_lower_arm"
  child: "right_hand"
  parent_offset {
    z: -0.258947
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 10.0
}
joints {
  name: "left_shoulder"
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    x: -0.02405
    y: 0.18311
    z: 0.2435
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 40.0
}
joints {
  name: "left_elbow"
  parent: "left_upper_arm"
  child: "left_lower_arm"
  parent_offset {
    z: -0.274788
  }
  child_offset {
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -160.0
    max: 0.0
  }
  angular_damping: 30.0
}
joints {
  name: "right_hip"
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.084887
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 50.0
}
joints {
  name: "right_knee"
  parent: "right_thigh"
  child: "right_shin"
  parent_offset {
    z: -0.421546
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angle_limit {
    min: 0.0
    max: 160.0
  }
  angular_damping: 50.0
}
joints {
  name: "right_ankle"
  parent: "right_shin"
  child: "right_foot"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 40.0
}
joints {
  name: "left_hip"
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.084887
  }
  child_offset {
  }
  rotation {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 50.0
}
joints {
  name: "left_knee"
  parent: "left_thigh"
  child: "left_shin"
  parent_offset {
    z: -0.421546
  }
  child_offset {
  }
  rotation {
    z: 90.0
  }
  angle_limit {
    min: 0.0
    max: 160.0
  }
  angular_damping: 50.0
}
joints {
  name: "left_ankle"
  parent: "left_shin"
  child: "left_foot"
  parent_offset {
    z: -0.40987
  }
  child_offset {
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
  angular_damping: 40.0
}
actuators {
  name: "spine1"
  joint: "spine1"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "neck"
  joint: "neck"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "right_shoulder"
  joint: "right_shoulder"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "right_elbow"
  joint: "right_elbow"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "right_wrist"
  joint: "right_wrist"
  strength: 50.0
  angle {
  }
}
actuators {
  name: "left_shoulder"
  joint: "left_shoulder"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "left_elbow"
  joint: "left_elbow"
  strength: 70.0
  angle {
  }
}
actuators {
  name: "right_hip"
  joint: "right_hip"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "right_knee"
  joint: "right_knee"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "right_ankle"
  joint: "right_ankle"
  strength: 90.0
  angle {
  }
}
actuators {
  name: "left_hip"
  joint: "left_hip"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "left_knee"
  joint: "left_knee"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "left_ankle"
  joint: "left_ankle"
  strength: 90.0
  angle {
  }
}
collide_include {
  first: "floor"
  second: "right_foot"
}
collide_include {
  first: "floor"
  second: "left_foot"
}
friction: 1.0
gravity {
  z: -9.81
}
angular_damping: -0.05
dynamics_mode: "pbd"
dt: 0.0333
substeps: 16
"""