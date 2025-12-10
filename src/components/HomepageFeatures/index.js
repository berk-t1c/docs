import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Ultra-Low Power',
    icon: '⚡',
    description: (
      <>
        Deliver vision systems at &lt;2W power consumption, enabling real-time AI 
        where GPUs physically fail. Process 10,000 fps equivalent performance 
        with minimal energy requirements.
      </>
    ),
  },
  {
    title: 'Sub-Millisecond Latency',
    icon: '🚀',
    description: (
      <>
        Achieve ~1ms response times for mission-critical applications. 
        Perfect for tactical edge scenarios requiring instant decision-making 
        without cloud dependency.
      </>
    ),
  },
  {
    title: 'High Dynamic Range',
    icon: '📷',
    description: (
      <>
        Capture 120dB dynamic range, enabling simultaneous tracking of objects 
        in bright and dark environments. Track satellites at 17,000 mph without 
        motion blur or saturation.
      </>
    ),
  },
  {
    title: 'Event-Driven Processing',
    icon: '🧠',
    description: (
      <>
        Process only changed pixels instead of full frames. Event cameras fire 
        only when pixels detect change, dramatically reducing computational overhead 
        and power consumption.
      </>
    ),
  },
  {
    title: 'Neuromorphic Computing',
    icon: '🔬',
    description: (
      <>
        Brain-inspired processors optimized for spiking neural networks. 
        Native spike processing eliminates frame reconstruction overhead, 
        enabling true event-driven AI.
      </>
    ),
  },
  {
    title: 'Edge AI Ready',
    icon: '🌐',
    description: (
      <>
        Zero cloud dependency for mission-critical systems. Deploy AI at the edge 
        with deterministic, low-latency inference suitable for defense, space, 
        and autonomous systems.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--4 margin-bottom--lg')}>
      <div className="text--center padding-horiz--md">
        <div className={styles.featureIcon}>{icon}</div>
        <Heading as="h3" className="margin-top--md">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className="text--center margin-bottom--xl">
              Key Capabilities
            </Heading>
          </div>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
