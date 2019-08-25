# AnimalAI カステム Unity環境

## 変更点

`BrainInfo`の`vector_observations`はデフォルトだとlocal velocityの情報(vx, vy, vz)だけが入っているが、これにAgentの位置情報と角度情報を付加する.

### TrainingAgent.cs

`Assets/AnimalAIOlympics/TrainEnv/Scripts/TrainingAgent.cs`

Globalの位置(x,y,z)と角度(0〜360)を追加で付加.


```
    public override void CollectObservations()
    {
        Vector3 localVel = transform.InverseTransformDirection(_rigidBody.velocity);
        AddVectorObs(localVel);

        // CHANGED: Custumly added global positaion and rotation
        Vector3 position = _rigidBody.position;
        AddVectorObs(position);

        float rotation = transform.eulerAngles.y;
        AddVectorObs(rotation);
    }
```

### BrainのParameter

`TrainEnv/Brains/Learner.asset`と`TrainEnv/Brains/Player.asset`

の`VectorObservations/SpaceSize`を`3`から`7`に変更


### 起動の高速化

`BuildSetting` -> `PlayerSettings` -> `SplashImage` -> `Splash Screen` -> `Show Splash Screen`のチェックを外すことで起動を高速化した.
(ここはUnity Proのライセンスでのみ設定可能)


